### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1000001-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["PlutoUI", "Plots", "CSV", "DataFrames", "FFTW", "Statistics"])
	using PlutoUI
	using Plots
	using CSV
	using DataFrames
	using FFTW
	using Statistics
end

# ╔═╡ a1000002-0000-0000-0000-000000000002
md"""
# Butterfly Flight Kinematics — Integrated Tracker

This notebook integrates the full tracking pipeline: OpenCV-based point tracking, kinematic analysis with FFT, and an interactive 3D model viewer.
"""

# ╔═╡ a1000003-0000-0000-0000-000000000003
md"""
## 1. Run Tracking Pipeline
Set the video path and click the button to run the OpenCV tracker. Check "Show Live" to see the tracking window on your screen.
"""

# ╔═╡ a1000004-0000-0000-0000-000000000004
@bind video_path TextField(default="data/raw/morpho_peleides.mp4")

# ╔═╡ a1000005-0000-0000-0000-000000000005
@bind show_live CheckBox(default=true)

# ╔═╡ a1000006-0000-0000-0000-000000000006
@bind run_btn Button("Run Full Pipeline")

# ╔═╡ a1000007-0000-0000-0000-000000000007
begin
	run_btn
	local status = "Ready. Click 'Run Full Pipeline' to start."
	if run_btn
		local cmd = show_live ? `python3 run_pipeline.py $video_path --live` : `python3 run_pipeline.py $video_path`
		run(cmd)
		status = "Pipeline finished. Data updated below."
	end
	status
end

# ╔═╡ a1000008-0000-0000-0000-000000000008
md"""
## 2. Kinematic Analysis
Wing area dynamics with adjustable smoothing, and FFT frequency spectrum.
"""

# ╔═╡ a1000009-0000-0000-0000-000000000009
md"Smoothing window:"

# ╔═╡ a100000a-0000-0000-0000-000000000010
@bind smooth_window Slider(3:2:31, default=11, show_value=true)

# ╔═╡ a100000b-0000-0000-0000-000000000011
begin
	run_btn  # reactivity trigger: re-run after pipeline
	
	local kpath = "output/combined/tracking/kinematics_per_frame.csv"
	if isfile(kpath)
		local kdf = CSV.read(kpath, DataFrame)
		
		# Moving average smoothing
		local halfwin = div(smooth_window - 1, 2)
		local smoothed = [mean(kdf.wing_proxy_area[max(1,i-halfwin):min(end,i+halfwin)]) for i in 1:nrow(kdf)]
		
		local p1 = plot(kdf.time_s, kdf.wing_proxy_area,
			label="Raw", color=:lightgrey, alpha=0.5,
			xlabel="Time (s)", ylabel="Area (px²)",
			title="Wing Area Dynamics")
		plot!(p1, kdf.time_s, smoothed, label="Smoothed (w=$smooth_window)", color=:steelblue, lw=2)
		
		# FFT
		local y = kdf.wing_proxy_area .- mean(kdf.wing_proxy_area)
		local fs = 1.0 / mean(diff(kdf.time_s))
		local N = length(y)
		local freqs = range(0, stop=fs/2, length=div(N,2))
		local mag = abs.(fft(y)[1:div(N,2)])
		
		local p2 = plot(freqs, mag,
			xlabel="Frequency (Hz)", ylabel="Magnitude",
			title="Wingbeat Frequency Spectrum (FFT)",
			label="FFT", color=:indianred, lw=1.5)
		xlims!(p2, 0, 30)
		
		local peak_i = argmax(mag[2:end]) + 1
		local peak_f = freqs[peak_i]
		vline!(p2, [peak_f], label="Peak: $(round(peak_f, digits=2)) Hz", ls=:dash, color=:orange)
		
		plot(p1, p2, layout=(2,1), size=(800, 650))
	else
		md"No data yet. Run the pipeline above first."
	end
end

# ╔═╡ a100000c-0000-0000-0000-000000000012
md"""
## 3. Raw Data Table
First 50 rows of the kinematics output. Use the download link to export the full CSV.
"""

# ╔═╡ a100000d-0000-0000-0000-000000000013
begin
	local kpath2 = "output/combined/tracking/kinematics_per_frame.csv"
	if isfile(kpath2)
		first(CSV.read(kpath2, DataFrame), 50)
	else
		md"No data yet."
	end
end

# ╔═╡ a100000e-0000-0000-0000-000000000014
begin
	local kpath3 = "output/combined/tracking/kinematics_per_frame.csv"
	if isfile(kpath3)
		DownloadButton(read(kpath3), "kinematics_data.csv")
	else
		md""
	end
end

# ╔═╡ a100000f-0000-0000-0000-000000000015
md"""
## 4. Interactive 3D Model
The reconstructed wing model with data-driven animation. Make sure the HTTP server is running on port 8765.
"""

# ╔═╡ a1000010-0000-0000-0000-000000000016
HTML("""
<div style="width:100%;height:620px;border:2px solid #444;border-radius:8px;overflow:hidden;background:#1e1e1e;">
	<iframe src="http://localhost:8765/view_3d_model.html" style="width:100%;height:100%;border:none;"></iframe>
</div>
""")

# ╔═╡ Cell order:
# ╠═a1000001-0000-0000-0000-000000000001
# ╟─a1000002-0000-0000-0000-000000000002
# ╟─a1000003-0000-0000-0000-000000000003
# ╠═a1000004-0000-0000-0000-000000000004
# ╠═a1000005-0000-0000-0000-000000000005
# ╠═a1000006-0000-0000-0000-000000000006
# ╠═a1000007-0000-0000-0000-000000000007
# ╟─a1000008-0000-0000-0000-000000000008
# ╟─a1000009-0000-0000-0000-000000000009
# ╠═a100000a-0000-0000-0000-000000000010
# ╠═a100000b-0000-0000-0000-000000000011
# ╟─a100000c-0000-0000-0000-000000000012
# ╠═a100000d-0000-0000-0000-000000000013
# ╠═a100000e-0000-0000-0000-000000000014
# ╟─a100000f-0000-0000-0000-000000000015
# ╠═a1000010-0000-0000-0000-000000000016
