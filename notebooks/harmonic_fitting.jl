#!/usr/bin/env julia
#=
harmonic_fitting.jl — Two/three-harmonic model fitting for wing kinematics.

Fits:  θ(t) = A·sin(2π·f·t) + h·sin(4π·f·t + φ) + offset

Usage:
  julia harmonic_fitting.jl data.csv            # fit from CSV
  julia harmonic_fitting.jl --test              # run self-test
  julia harmonic_fitting.jl data.csv --3harm    # three-harmonic model

CSV must have columns: time_s, angle_rad (or specify --time-col / --angle-col)
=#

using Pkg

for dep in ["LsqFit", "CSV", "DataFrames", "FFTW"]
    if !haskey(Pkg.project().dependencies, dep)
        Pkg.add(dep)
    end
end

using LsqFit
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Printf
using DelimitedFiles
using FFTW


# ──────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────

# Two-harmonic: θ(t) = A·sin(2π·f·t) + h·sin(4π·f·t + φ) + offset
# Parameters p = [A, f, h, φ, offset]
#
# Broadcasting: every op on vector t uses dot-syntax (.*, .+, sin.())
# since t is a vector and p contains scalars. Forgetting dots
# causes DimensionMismatch — the most common Julia error here.
function model_2harm(t, p)
    A, f, h, φ, offset = p
    return A .* sin.(2π .* f .* t) .+ h .* sin.(4π .* f .* t .+ φ) .+ offset
end

# Three-harmonic: adds A₃·sin(6π·f·t + φ₃)
# Parameters p = [A, f, h, φ, offset, A3, φ3]
function model_3harm(t, p)
    A, f, h, φ, offset, A3, φ3 = p
    return (A .* sin.(2π .* f .* t)
          .+ h .* sin.(4π .* f .* t .+ φ)
          .+ A3 .* sin.(6π .* f .* t .+ φ3)
          .+ offset)
end


# ──────────────────────────────────────────────────────────
#  Initial guess from FFT
# ──────────────────────────────────────────────────────────

function initial_guess(t::Vector{Float64}, θ::Vector{Float64})
    N = length(θ)
    dt = mean(diff(t))
    fs = 1.0 / dt

    offset = mean(θ)
    centered = θ .- offset

    # FFT to find dominant frequency
    Y = abs.(rfft(centered))
    freqs = range(0, stop=fs/2, length=length(Y))

    # skip DC (index 1), find peak
    peak_idx = argmax(Y[2:end]) + 1
    f_guess = freqs[peak_idx]

    A_guess = (maximum(θ) - minimum(θ)) / 2.0
    h_guess = A_guess * 0.25  # second harmonic typically 20-30% of fundamental
    φ_guess = 0.0

    return [A_guess, f_guess, h_guess, φ_guess, offset]
end


# ──────────────────────────────────────────────────────────
#  Fitting
# ──────────────────────────────────────────────────────────

function fit_harmonics(t::Vector{Float64}, θ::Vector{Float64};
                       use_3harm::Bool=false)

    # Normalize time to [0,1] — reduces parameter correlation.
    # Large t values make f and φ highly correlated, which stalls the optimizer.
    t_min, t_max = extrema(t)
    t_range = t_max - t_min
    if t_range == 0.0
        t_range = 1.0
    end
    t_norm = (t .- t_min) ./ t_range

    p0 = initial_guess(t_norm, θ)
    # p0[2] is already in normalized-time frequency units (since initial_guess
    # runs on t_norm), so no rescaling needed here.

    if use_3harm
        p0 = vcat(p0, [p0[1] * 0.1, 0.0])  # A3 ≈ 10% of A, φ3 = 0
        model = model_3harm
    else
        model = model_2harm
    end

    # curve_fit expects model(xdata, params) → vector (not scalar)
    fit = curve_fit(model, t_norm, θ, p0; maxIter=1000)

    params = fit.param

    # Convert frequency back to real time
    params_real = copy(params)
    params_real[2] /= t_range

    # 95% confidence intervals
    ci = try
        confidence_interval(fit, 0.05)
    catch
        [(NaN, NaN) for _ in params]
    end

    predicted = model(t_norm, params)
    resid = θ .- predicted
    ss_res = sum(resid .^ 2)
    ss_tot = sum((θ .- mean(θ)) .^ 2)
    r² = 1.0 - ss_res / ss_tot
    rms = sqrt(mean(resid .^ 2))

    # Durbin-Watson for residual autocorrelation (≈2 means no autocorrelation)
    dw = sum(diff(resid) .^ 2) / ss_res

    return (
        params = params_real,
        params_normalized = params,
        fit = fit,
        ci = ci,
        r_squared = r²,
        residuals = resid,
        rms_error = rms,
        durbin_watson = dw,
        t_norm_range = (t_min, t_range),
        model_type = use_3harm ? "3-harmonic" : "2-harmonic",
    )
end


# ──────────────────────────────────────────────────────────
#  Evaluation & printing
# ──────────────────────────────────────────────────────────

function evaluate_model(result, t_eval::Vector{Float64})
    t_min, t_range = result.t_norm_range
    t_norm = (t_eval .- t_min) ./ t_range
    if result.model_type == "3-harmonic"
        return model_3harm(t_norm, result.params_normalized)
    else
        return model_2harm(t_norm, result.params_normalized)
    end
end

function model_string(result)
    p = result.params
    if result.model_type == "3-harmonic"
        A, f, h, φ, offset, A3, φ3 = p
        return @sprintf("θ(t) = %.4f·sin(2π·%.4f·t) + %.4f·sin(4π·%.4f·t + %.4f) + %.4f·sin(6π·%.4f·t + %.4f) + %.4f",
                        A, f, h, f, φ, A3, f, φ3, offset)
    else
        A, f, h, φ, offset = p
        return @sprintf("θ(t) = %.4f·sin(2π·%.4f·t) + %.4f·sin(4π·%.4f·t + %.4f) + %.4f",
                        A, f, h, f, φ, offset)
    end
end

function print_results(result)
    println("\n" * "="^60)
    println("  Harmonic Fit Results ($(result.model_type))")
    println("="^60)

    names_2 = ["A (amplitude)", "f (frequency Hz)", "h (2nd harmonic amp)",
               "φ (2nd harmonic phase)", "offset"]
    names_3 = vcat(names_2, ["A₃ (3rd harmonic amp)", "φ₃ (3rd harmonic phase)"])
    param_names = result.model_type == "3-harmonic" ? names_3 : names_2

    for (i, name) in enumerate(param_names)
        lo, hi = result.ci[i]
        @printf("  %-25s = %10.6f   (95%% CI: [%.6f, %.6f])\n",
                name, result.params[i], lo, hi)
    end

    println()
    @printf("  R²              = %.6f\n", result.r_squared)
    @printf("  RMS residual    = %.6f rad (%.4f°)\n",
            result.rms_error, rad2deg(result.rms_error))
    @printf("  Durbin-Watson   = %.4f  (≈2 = no autocorrelation)\n",
            result.durbin_watson)
    println()
    println("  Model: ", model_string(result))
    println("="^60)
end


# ──────────────────────────────────────────────────────────
#  Export
# ──────────────────────────────────────────────────────────

function export_params_csv(result, path::String)
    open(path, "w") do f
        write(f, "parameter,value,ci_lower,ci_upper\n")
        names = result.model_type == "3-harmonic" ?
            ["A", "f", "h", "phi", "offset", "A3", "phi3"] :
            ["A", "f", "h", "phi", "offset"]
        for (i, name) in enumerate(names)
            lo, hi = result.ci[i]
            write(f, "$name,$(result.params[i]),$lo,$hi\n")
        end
        write(f, "\n# R² = $(result.r_squared)\n")
        write(f, "# RMS = $(result.rms_error)\n")
        write(f, "# Model: $(model_string(result))\n")
    end
    println("  Saved parameters → $path")
end

function export_timeseries_csv(result, t_start::Float64, t_end::Float64,
                                path::String; n_points::Int=10000)
    t_dense = collect(range(t_start, stop=t_end, length=n_points))
    θ_dense = evaluate_model(result, t_dense)

    open(path, "w") do f
        write(f, "time_s,theta_rad,theta_deg\n")
        for i in eachindex(t_dense)
            @printf(f, "%.8f,%.8f,%.4f\n",
                    t_dense[i], θ_dense[i], rad2deg(θ_dense[i]))
        end
    end
    println("  Saved $(n_points) points → $path")
end

function export_julia_function(result, path::String)
    p = result.params
    open(path, "w") do f
        write(f, "# Fitted wing stroke angle model\n")
        write(f, "# $(model_string(result))\n")
        write(f, "# R² = $(result.r_squared)\n\n")

        if result.model_type == "3-harmonic"
            A, freq, h, φ, offset, A3, φ3 = p
            write(f, """function theta(t)
    return $A * sin(2π * $freq * t) + $h * sin(4π * $freq * t + $φ) + $A3 * sin(6π * $freq * t + $φ3) + $offset
end
""")
        else
            A, freq, h, φ, offset = p
            write(f, """function theta(t)
    return $A * sin(2π * $freq * t) + $h * sin(4π * $freq * t + $φ) + $offset
end
""")
        end
    end
    println("  Saved Julia function → $path")
end

function export_matlab_function(result, path::String)
    p = result.params
    open(path, "w") do f
        write(f, "% Fitted wing stroke angle model\n")
        write(f, "% $(model_string(result))\n")
        write(f, "% R^2 = $(result.r_squared)\n\n")
        write(f, "function theta = wing_theta(t)\n")

        if result.model_type == "3-harmonic"
            A, freq, h, φ, offset, A3, φ3 = p
            write(f, "    theta = $A.*sin(2*pi*$freq.*t) + $h.*sin(4*pi*$freq.*t + $φ) + $A3.*sin(6*pi*$freq.*t + $φ3) + $offset;\n")
        else
            A, freq, h, φ, offset = p
            write(f, "    theta = $A.*sin(2*pi*$freq.*t) + $h.*sin(4*pi*$freq.*t + $φ) + $offset;\n")
        end
        write(f, "end\n")
    end
    println("  Saved MATLAB function → $path")
end


# ──────────────────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────────────────

function run_self_test()
    println("\n  Running self-test...")

    A_true  = 1.2
    f_true  = 12.0
    h_true  = 0.3
    φ_true  = 0.5
    off_true = 0.1

    # 120 fps, 2 seconds
    fps = 120.0
    t = collect(0.0 : 1/fps : 2.0)
    θ_clean = A_true .* sin.(2π .* f_true .* t) .+
              h_true .* sin.(4π .* f_true .* t .+ φ_true) .+
              off_true

    # Realistic tracking noise (~1°)
    θ_noisy = θ_clean .+ 0.02 .* randn(length(t))

    result = fit_harmonics(t, θ_noisy)
    print_results(result)

    A_fit, f_fit, h_fit, φ_fit, off_fit = result.params

    passed = true
    checks = [
        ("A",      A_fit,   A_true,   0.1),
        ("f",      f_fit,   f_true,   0.5),
        ("h",      h_fit,   h_true,   0.1),
        ("φ",      φ_fit,   φ_true,   0.3),
        ("offset", off_fit, off_true, 0.05),
    ]

    println("\n  Parameter recovery:")
    for (name, fitted, true_val, tol) in checks
        err = abs(fitted - true_val)
        ok = err < tol
        status = ok ? "PASS" : "FAIL"
        @printf("    %-8s: fitted=%.4f  true=%.4f  error=%.4f  [%s]\n",
                name, fitted, true_val, err, status)
        if !ok
            passed = false
        end
    end

    @printf("\n  R² = %.6f (should be > 0.99)\n", result.r_squared)
    if result.r_squared < 0.99
        passed = false
    end

    println("\n  Self-test: ", passed ? "ALL PASSED" : "SOME FAILED")
    return passed
end


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

function main()
    if "--test" in ARGS
        run_self_test()
        return
    end

    if length(ARGS) < 1
        println("Usage:")
        println("  julia harmonic_fitting.jl <data.csv> [--3harm] [--time-col name] [--angle-col name]")
        println("  julia harmonic_fitting.jl --test")
        return
    end

    csv_path = ARGS[1]
    use_3harm = "--3harm" in ARGS

    time_col = "time_s"
    angle_col = "angle_rad"
    for i in 1:length(ARGS)-1
        if ARGS[i] == "--time-col"
            time_col = ARGS[i+1]
        elseif ARGS[i] == "--angle-col"
            angle_col = ARGS[i+1]
        end
    end

    println("  Loading $csv_path...")
    df = CSV.read(csv_path, DataFrame; comment="#")

    if !(time_col in names(df))
        println("  Error: column '$time_col' not found. Available: $(names(df))")
        return
    end
    if !(angle_col in names(df))
        println("  Error: column '$angle_col' not found. Available: $(names(df))")
        return
    end

    t = Float64.(df[!, time_col])
    θ = Float64.(df[!, angle_col])

    valid = .!isnan.(t) .& .!isnan.(θ)
    t = t[valid]
    θ = θ[valid]

    println("  Loaded $(length(t)) points, t ∈ [$(minimum(t)), $(maximum(t))] s")

    result = fit_harmonics(t, θ; use_3harm=use_3harm)
    print_results(result)

    base = splitext(csv_path)[1]
    export_params_csv(result, base * "_harmonics.csv")
    export_timeseries_csv(result, minimum(t), maximum(t), base * "_fitted.csv")
    export_julia_function(result, base * "_theta.jl")
    export_matlab_function(result, base * "_theta.m")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
