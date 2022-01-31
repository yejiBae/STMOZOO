### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a5d4303a-81b0-11ec-1083-ed13a1d29807
using PlutoUI, LinearAlgebra, FFTW,Plots, StructuredOptimization, Distributions, FileIO, QuartzImageIO, ImageShow, TestImages , Colors

# ╔═╡ f4cb8cfd-b6b1-4e93-9985-2c71b384093f
md"""
# Recovering images from a few pixels using compressed sensing (L₁-norm minimization)

Project by Yeji Bae and Manon Montyne

## Introduction

### Data compression 

In data compression, a certain signal is represented more efficiently in terms of the sparse vector of coefficients in a generic transform basis e.g. Fourier or wavelet bases. Since most natural signals are compressible, they can be written in terms of a sparse vector s ∈ Rⁿ and a transform basis Ψ ∈ Rⁿ ˣ ⁿ	:

									x = Ψs

The number of non-zero elements in the vector s equals K. Therefore, we also describe s as being K-sparse.

Both images and audio signals are compressible in Fourier or wavelet bases. After taking the transform, most coefficients are close to zero and can thus be set equal to zero with a limited loss of quality. Afterward, these few active coefficients can replace the original signal to be stored and transmitted. The original signal can then be recovered by taking the inverse transform [1].


Example of data compression (adapted code from [1]):
"""

# ╔═╡ 62a8366f-14dc-410b-ba31-a8f695dfe986
begin
	url = "https://www.thespruce.com/thmb/JObhDcNNBALGVyF9S7bFoQwoXsI=/941x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/plants-with-big-flowers-4138211-hero-b10becb169064cc4b3c7967adc1b22e1.jpg"
	download(url, "flower.jpeg")
	flower = load("flower.jpeg") #Load in mage from url
end

# ╔═╡ 97006203-2819-489d-a4dc-cef3af6e72b3
flower2 = Gray.(flower) #Convert to grayscale

# ╔═╡ 7415fc09-1155-49e5-b523-c9df2bd8d7d7
begin
	flower2_array = convert(Array{Float64}, flower2) #Convert image to array
	flower_tranformed = fft(flower2_array) #Apply Fourier transformation
	F = log.(abs.(fftshift(flower_tranformed)) .+1) #Put on log-scale
	Gray.(F) #Plot Fourrier coefficients
end

# ╔═╡ bc36eb16-5b31-4639-864c-4ee4b627108e
begin
	Bt = sort(abs.(flower_tranformed[:,:]), dims=2) #sortFourrier coefficients
	thr= floor.(0.95*length.(Bt)) #Keep 5% of the data (1-0.95 = 0.05)
	ind = abs.(flower_tranformed) .> thr
	Atlow = flower_tranformed.*ind
	alow = ifft(Atlow)	#Take inverse FFT (iFFT) to plot compressed image
	Gray.(abs.(alow))
end

# ╔═╡ 56a83328-2689-42be-8dfd-8ee38c5a70e0
md"""
As can be seen in this example, we were able to reconstruct the original image by keeping only 5% of the data.

Compressed sensing on the other hand, turns the compression paradigm upside down: instead of compressing high-dimensional data by discarding most of the information, it is now possible to reconstruct a full signal from only a few measurements. Before we go deeper into compressed sensing, we will have a look at the reconstruction of a simple signal (based on an example from Kutz et al. (7)).

### Reconstruction of a simple signal (Kutz example)

This example is based on the Kutz example which was worked out further in Nextjournal (6). Herein, a simple signal is constructed from 2 frequencies.

					y = sin.(1394 * π * t) + sin.(3266 * π * t)

In the following plots we show the signal both in the time domain and the frequency domain. These plots clearly show that in the time domain the signal is not sparse, whereas in the frequency domain it is. By looking at the frequency domain, we see that the signal is zero everywhere except at two points, representing th etwo cosine waves. This example thus introduces 2 very important concepts in compressed sensing: sparsity and incoherence.
"""

# ╔═╡ 6e37dfac-5f50-498b-b356-f56fcbd7d02a
begin
	
	n = 5000 #number of samples
	t = range(0.0, step=1/n, length=n) #Step size is 1/5000
	y = sin.(1394 * pi * t) + sin.(3266 * pi * t) #Signal consisting of 2 sinus waves (Kutz example)
	plot(t, y, title = "Simple signal (time domain)", xlabel="t", ylabel="y", legend=false)
end

# ╔═╡ 68a5c2c4-e1ca-420d-b421-c4f570308884
plot(t, y, xlim=(0.0, 0.02), title = "Zoom in on simple signal plot (time domain)", xlabel = "t", ylabel = "y", legend=false)

# ╔═╡ 39f25941-cab4-4776-9167-5b75d555fdbf
begin
	freq = fftfreq(n,n)
	plot(freq,dct(y),title="Simple signal (frequency domain)", legend=false, xlabel = "Frequency", ylabel = "dct(y)") #dct = discrete cosine transform which is needed to get the frequency domain signal (a widely used transformation technique in signal processing and data compression)
end

# ╔═╡ c899003d-bee5-4857-919a-8007d22c9f90
md"""
## Compressed sensing

### Sparsity
All of data compression relies on sparsity. This means that a signal is represented more efficiently in terms of a sparse vector of coefficients in a transform basis (e.g. Fourier or wavelet basis). As previously seen, signals can thus be written in terms of a sparse vector s ∈ Rⁿ and a transform basis Ψ ∈ Rⁿ ˣ ⁿ:

									x = Ψs [Eq. 1]

Now, in compressed sensing it is possible to collect only a few measurements and then infer the sparse representation in the transformed basis. Until recently, finding the sparsest vector consistent with the measurements was considered a NP hard problem. However, nowadays it is possible to achieve a full signal reconstruction using convex optimization algorithms.

If we have a signal that is K-sparse in Ψ, it is possible to collect only a few measurements y ∈ Rᵖ, with K < p << n (with n = number of measurements in original signal x). These measurements y are then given by:

									y = Cx [Eq. 2]

In this formula, C ∈ Rᵖ ˣ ⁿ is the measurement matrix  with a set of p linear measurements on the state of x. The choice of this measurement matrix is very important in compressed sensing because it must be incoherent with respect to Ψ (see further). 

Another important condition for compressed sensing is that the number of measurements p must be sufficiently large. It must be in the order of p =  (K * log(n/K)) = k₁ * K * log(n/K) (k₁ depends on how incoherent C and Ψ are)

If we now know the sparse vector s, we are capable of reconstructing the signal x. The ultimate goal in compressed sensing is thus to find the sparsest vector s that is consistent with the measurements. 

								y = CΨs = Θs [Eq. 3]
This system of equations is an underdetermined problem and we thus have infinitely many solutions for s, however, we want to find the sparsest s for which these equations hold:
					ŝ = argmin ||s||₀ subject to y = CΨs [Eq. 4]

This optimization however is non-convex and a solution can only be found by brute-force search, therefore we will relax the optimization to a convex optimization by taking the l₁ norm or Manhattan norm:

					ŝ = argmin ||s||₁ subject to y = CΨs [Eq. 5]
[1]


*Note on different norms*
- *The l₀ norm is used to count non-zero elements in a vector*
- *The l₁ norm sums the absolute values of each element in a vector*


### Incoherence

We have now discussed the concept of sparsity and compressed sensing. In this part we will discuss incoherence, because an important part of compressed sensing is the incoherence of the measurement matrix C (see before).

If we have enough good measurements we will have a matrix Θ = CΨ that preserves the distance and inner product structure of sparse vectors. We thus seek a measurement matrix C such that Θ acts as a near isometry map on sparse vectors. Isometry means that we perform a distance-preservind transformation. If now Θ acts as a near isometry, we can solve equation 3 using convex l₁ minimization.

In other words, the measurements C are good measurements if they are incoherent with respect to the columns of the sparsifying basis Ψ. In this case the rows of C will have a small inner product with the columns of Ψ. On the other hand, if the measurements are coherent with the basis, it will provide little information. The (in)coherence of measurements C and the basis Ψ is given by $$μ(C,Ψ) = \sqrt{n}*max|⟨c_{k},ψ_{j}⟩|$$ where $$c_{k}$$ is the k-th row of the matrix C and $$ψ_{j}$$ is the j-th column of the matrix ψ. The coherence will range between 1 and $$\sqrt{n}$$ [1].

Hence, in compressed sensing it is important to find low coherence pairs. A first example of such a pair is the canonical or spike basis ϕₖ(t) = δ(t − k) as C combined with the Fourier basis as Ψ. This pair has a μ(C,Ψ) = 1, giving us maximal incoherence. In a second example, we have noiselets for C and wavelet bases as Ψ. The coherence between this pair is $$\sqrt{2}$$. Finally, also random matrices are largely incoherent with any fixed basis. For this purpose, an orthobasis C needs to be selected uniformly at random. This can be done by orthonormalizing n vectors sampled independently and uniformly on the unit sphere. The coherence between C and Ψ in this case is then about 2 log(n) [8]. This last example is shown below in the code.
"""

# ╔═╡ 135c1b29-8e82-4d06-b2f6-ff6fa7e3f031
md"""
### Kutz example part 2: reconstructing the signal from a sample

Now we will work further on the Kutz example. In the following plot, we sampled 10% of the original signal and made a scatter plot of the data points. The data points were also connected through lines. The percentage of sampling can be changed with the following slider. The code that follows was adapted from Nextjournal [6].

$@bind x Slider(1:100, show_value=true, default=10)
"""

# ╔═╡ 832052d7-c34d-4bce-8236-64cdc8f32be5
begin
	#Sample x procent of the signal
	n_sample= Int(n*x/100) #number of data points sampled
	sampled_t_coordinates = sort(rand(1:n,n_sample)) # sorting for nice plotting
	#Sample the signal
	sampled_y_coordinates = y[sampled_t_coordinates]
	#Plot sampled signal
	plot(sampled_t_coordinates, sampled_y_coordinates, seriestype=scatter, title = "Sparse signal", xlabel="t", ylabel="y", legend=false) #Scatter
	plot!(sampled_t_coordinates,sampled_y_coordinates,seriestype=:line) #Add line
end

# ╔═╡ df23221a-1dc6-4f45-9f03-35bc12c48e46
md"""

Now we have our sparse data and we would like to reconstruct our original signal. As previously stated, we can solve this problem using convex optimization:

					ŝ = argmin ||s||₁ subject to y = CΨs

We can solve this using the StructuredOptimization.jl package (more details on this package later).
"""

# ╔═╡ 7a9864b2-fc6f-4b89-8efa-183d644f8943
begin
	s = Variable(n)
	~s .= 0.0
	lambda = 1e-1
	@minimize ls(idct(s)[sampled_t_coordinates]-sampled_y_coordinates) + lambda*norm(s,1)
end

# ╔═╡ d32b08c7-8262-4231-9469-df9f667e5c59
~s

# ╔═╡ c490029b-2b61-42c4-bd5f-72ba844e03b0
begin
	plot(freq,dct(y),color="blue",title="Original vs reconstructed signal (frequency domain)", label = "Original signal", xlabel = "Frequency", ylabel = "dct(y)")
	plot!(freq,~s,color="orange", label = "Reconstructed signal")
end

# ╔═╡ 7a104d94-6062-47a2-82cb-553513e28a86
begin
	plot(t, y,xlim=(0.0, 0.02), title = "Original vs reconstructed signal (time domain)", xlabel = "t", ylabel = "y", color="blue", label = "Original signal")
	plot!(t, idct(~s), label = "Reconstructed signal", color="orange")
end

# ╔═╡ 2bef7c97-8b88-4203-8825-eda70d5fab56
md"""
This result is pretty good for only 10 percent of data. The higher the percentage of sampling (adjust with the slider), the more related the original and the reconstructed signal become, as expected.

Now, to show that we cannot use l₂-norm minimization to solve the sparsity problem, we have solved the example for the l₂-norm and plotted the results above the previous plots. We clearly see that the signal is not sparse in the frequency domain, in contrast to the l₁-norm. We are thus not able to reconstruct the original signal using l₂-norm minimization (see plot in time domain).
"""

# ╔═╡ 9c19aaf2-3c25-425b-9e48-4d90d1366ce2
begin
	s1 = Variable(n)
	~s1 .= 0.0
	@minimize ls(idct(s1)[sampled_t_coordinates]-sampled_y_coordinates) + lambda*norm(s1,2)
end

# ╔═╡ 4d4ec391-7219-459a-9b49-6ba901eeae92
~s1

# ╔═╡ 57a799e2-68b2-4867-867a-24d8590a84a5
begin
	plot(freq,dct(y),color="blue",title="Original vs reconstructed signal (frequency domain)", label = "Original signal", xlabel = "Frequency", ylabel = "dct(y)")
	plot!(freq,~s,color="orange", label = "Reconstructed signal (l₁)")
	plot!(freq, ~s1, color="green", label = "Reconstructed signal (l₂)")
end

# ╔═╡ 7c3a2e4b-419f-44f1-bea7-20c7a3aa06a3
begin
	plot(t, y,xlim=(0.0, 0.02), title = "Original vs reconstructed signal (time domain)", xlabel = "t", ylabel = "y", color="blue", label = "Original signal")
	plot!(t, idct(~s), label = "Reconstructed signal (l₁)", color="orange")
	plot!(t, idct(~s1), label = "Reconstructed signal (l₂)", color = "green")
end

# ╔═╡ 84e6d7fb-4df6-498b-bd5c-151004756e9e
md"""
### Application: MRI

Magnetic Resonance Imaging (MRI) is a medical imaging technique that is widely used to provide reproducible, non-invasive, and quantitative measurements of tissue. However, a major disadvantage of MRI is the long acquisition time. Due to this disadvantage, there could be major artifacts present when there is patient or physiological movement. This prolonged acquisition time also limits the usage of MRI due to its expensive cost and considerations of patient comfort and compliance. Furthermore, if contrast agent injection is required, there could be a contrast washout leading to poor quality or non-diagnostic images [2].

The reason for this slow acquisition time is that data samples are being collected in the k-space, rather than directly in the image space. This k-space is the 2D or 3D Fourier transform of the MR image that is measured. The raw data samples are obtained sequentially in this k-space and the speed at which this is done is limited by both physiological and hardware constraints. Once the desired field-of-view and spatial resolution are specified, the required raw data in the k-space is conventionally determined by the Nyquist-Shannon sampling criteria. However, compressed sensing allows for fast acquisition without adhering to the Nyquist-Shannon sampling criteria by performing a more aggressive undersampling. With compressed sensing, an image can be reconstructed without deterioration of the quality by performing a non-linear optimization on randomly undersampled raw data, assuming that the data is compressible [2], [3].

For this purpose, it is important to find a good undersampling scheme that is as random as possible to create incoherent undersampling artifacts so that a proper non-linear reconstruction can be applied (see the part about incoherence). The acceleration rate is determined by the sparsity of this random undersampling. Secondly, we also need a good non-linear optimization algorithm to have an efficient, stable, and accurate reconstruction [2].
"""

# ╔═╡ afb35a7d-1454-4c8f-a230-daabafc8e7a3
md"""
### Algorithms for l₁-norm minimization

Nowadays, there are mainly 2 categories of reconstruction algorithms: convex optimization algorithms and greedy algorithms. The greedy algorithm uses a fast iterative process, whereas the convex optimization algorithms have a higher computational complexity [11].  In the StructuredOptimization.jl package that was used in the Kutz example, we used LASSO for l₁-minimization. However, other proximal algorithms can be used by adding a 'solver object' from the ProximalAlgorithms.jl package [10]. Least Absolute Shrinkage and Selection Operator (LASSO) is a penalizd least squares technique for which we can rewrite the problem in Eq. 5 as:

					ŝ = argmin (||CΨs - y||²)/2 + λ||s||₁  [Eq. 6]

where λ ≥ 0 is the regularization parameter which controls the amount of shrinkage. It makes us of the coordinate descent algorithm [12].

In the following paragraphs, we will discuss two commonly used algorithms to perform l₁-norm minimization. The first one is the Alternating Direction Method of Multipliers (ADMM) and the second one is l₁-norm minimization via l₂-norm minimization.

##### Alternating Direction Method of Multipliers (ADMM)

ADMM is an algorithm which solves convex optimization problems by breaking them into smaller subproblems, that are easier to solve. The solutions to these small local subproblems are then used to find the solution to the large global problem. ADMM blends the benefits of dual composition and augmented Lagrangian methods for constrained optimization and is closely related to a lot of other algorithm such as proximal methods . In general, ADMM is very efficient in optimization, however, it is not trivial to determine the optimal parameter settings (e.g. penalty parameters, update rates) that influence the accuracy and speed of reconstruction.

the algorithm solves problem in the form:
	
					minimize f(x) + g(z) subject to Ax + Bz = c

The only difference between this and the general linear equality-constrained problem:

					minimize f(x) subject to Ax = b

is that the variable, called x there, has here been split into 2 parts (x and z). The augmented Lagrangian for this problem is:

			Lρ(x, z, y) = f(x) + g(z) + yᵀ(Ax + Bz - c) + (ρ/2)||Ax + Bz - c||²₂

ADMM will then peform iterations in which there is an x-minimization step, a z-minimization step and a dual variable update (which is very similar to dual ascent method and method of multipliers). The dual variable update will use a step size that is equal to the augmented Lagrangian parameter ρ.

				xᵏ⁺¹ := argmin(x)  Lρ(x, zᵏ, yᵏ)
				zᵏ⁺¹ := argmin(z)  Lρ(xᵏ⁺¹, z, yᵏ)
				yᵏ⁺¹ := yᵏ + ρ(Axᵏ⁺¹ + Bzᵏ⁺¹ - c) where ρ > 0

In ADMM, x and z are thus updated in alternating or sequential fashion. This is reflected in the name by calling it 'alternating direction' method of multipliers [9].

##### l₁-norm minimization via l₂-norm minimization 

In this last paragraph we introduce the state of the art compressed sensing reconstruction algorithm which uses l₂-norm minimization to achieve l₁-norm minimization. This method makes use of the geometrical interpretation of the minimization problem and the fact that the Euclidean distance between the l₁-norm and the l₂-norm solutions lie closely to one another. This means that if we can find the l₂-norm solution, the direction of the l₁-norm solution is on the shortest path connecting them. This method provides a more stable solution than the greedy algorithm and is faster compared to the convex optimization methods. The pseudocode for the method is provided below:

*Pseudocode*

- Input: y ∈ Rᴹ and A ∈ Rᴹˣᴺ with sparsity k
- Output: xˆ ∈ Rᴺ
- Extra information: Q = l₂-norm solution, P = l₁-norm minimization

		1: [M, N] <- dim(A)  # initiate the dimension
		2: γ <- det(AAᵀ)
		3: If γ = 0 then stop, return xˆ<- {}, else continue to step 4
		4: calculate xQ <-(AAᵀ)-Aᵀy  # L2-norm solution
		5: [U ζ] = sort(abs(xQ))      # sort L2-norm solution 
		6: ds = zeros (N,1)           # initiate direction vector
		7: for i=1 to k
		8: ds(∫(k))=1    # vector direction from L2-norm solution to L1-norm solution
		9: end for loop
		10: As = Ads,
		11: xP = (AsAsᵀ)^(-1)Asᵀy    # L1-norm solution
		12: xˆ<- xp 
		13: return xˆ

"""

# ╔═╡ ef90c37f-0e7e-48d1-8ab3-4ae779c3ed91
md"""
## References

[1] S. L. Brunton and N. J. Kutz, Data-Driven Science & Engineering. 2017.

[2] G. Yang et al., “DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction,” IEEE Trans. Med. Imaging, vol. 37, no. 6, pp. 1310–1321, Jun. 2018.

[3] “k-space (magnetic resonance imaging) - Wikipedia.” [Online]. Available: https://en.wikipedia.org/wiki/K-space_(magnetic_resonance_imaging). [Accessed: 17-Dec-2021]

[4] M. Lustig, D. L. Donoho, J. M. Santos, and J. M. Pauly, “Compressed sensing MRI: A look at how CS can improve on current imaging techniques,” IEEE Signal Process. Mag., vol. 25, no. 2, pp. 72–82, 2008.

[5] Pereira, M. P., Lovisolo, L., da Silva, E. A. B., &amp; de Campos, M. L. R. (2014). On the design of maximally incoherent sensing matrices for compressed sensing using orthogonal bases and its extension for Biorthogonal Bases Case. Digital Signal Processing, 27, 12–22. https://doi.org/10.1016/j.dsp.2014.01.006 

[6] Compressed Sensing in Julia. Nextjournal. 2021 Feb 15

[7] Brunton, Steven L., and Jose N. Kutz. Data-driven science and engineering: machine learning, dynamical systems, and control. Cambridge, United Kingdom New York, NY: Cambridge University Press, 2019. Print.

[8] Candes EJ, Wakin MB. An introduction to compressive sampling: A sensing/sampling paradigm that goes against the common knowledge in data acquisition. IEEE Signal Process Mag [Internet]. 2008 [cited 2022 Jan 30];25(2):21–30. Available from: https://www.researchgate.net/publication/3322018_Wakin_MB_An_introduction_to_compressive_sampling_IEEE_Signal_Process_Mag_252_21-30

[9] Boyd S, Parikh N, Chu E, Eckstein J, Boyd S, Parikh N, et al. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Found Trends R Mach Learn. 2010;3(1):1–122. 

[10] Home · StructuredOptimization [Internet]. [cited 2022 Jan 31]. Available from: https://juliafirstorder.github.io/StructuredOptimization.jl/latest/

[11] Usman, K., Gunawan, H., &amp; Suksmono, A. B. (2018). Compressive Sensing Reconstruction Algorithm using L1-norm Minimization via L2-norm Minimization. International Journal on Electrical Engineering and Informatics, 10(1), 37–50. https://doi.org/10.15676/ijeei.2018.10.1.3 

[12] Gauraha N. Introduction to the LASSO A Convex Optimization Approach for High-dimensional Problems.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuartzImageIO = "dca85d43-d64c-5e67-8c65-017450d5d020"
StructuredOptimization = "46cd3e9d-64ff-517d-a929-236bc1a1fc9d"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
Colors = "~0.12.8"
Distributions = "~0.25.43"
FFTW = "~1.3.0"
FileIO = "~1.12.0"
ImageShow = "~0.3.3"
Plots = "~1.25.7"
PlutoUI = "~0.7.32"
QuartzImageIO = "~0.7.4"
StructuredOptimization = "~0.3.0"
TestImages = "~1.6.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "051c95d6836228d120f5f4b984dd5aba1624f716"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "0.5.0"

[[AbstractOperators]]
deps = ["AbstractFFTs", "DSP", "FFTW", "LinearAlgebra", "RecursiveArrayTools"]
git-tree-sha1 = "ea4e4ad8d7def7e6e83e2336454d67d4cc5c421f"
uuid = "d9c5613a-d543-52d8-9afd-8f241a8c3f1c"
version = "0.2.2"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "ffc6588e17bcfcaa79dfa5b4f417025e755f83fc"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "6b6f04f93710c71550ec7e16b650c1b9a612d0b6"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.16.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DSP]]
deps = ["FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "2a63cb5fc0e8c1f0f139475ef94228c7441dc7d0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.6.10"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "38bcc22b6e358e88a7715ad0db446dfd3a4fea47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.43"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d7ab55febfd0907b285fbf8dc0c73c0825d9d6aa"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.3.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "IntelOpenMP_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Reexport"]
git-tree-sha1 = "8fda0934cb99db617171f7296dc361f4d6fa5424"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.3.0"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "67551df041955cc6ee2ed098718c8fcd7fc7aebe"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.12.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JSONSchema]]
deps = ["HTTP", "JSON", "URIs"]
git-tree-sha1 = "2f49f7f86762a0fbbeef84912265a1ae61c4ef80"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.4"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["IntelOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "eb540ede3aabb8284cb482aa41d00d6ca850b1f8"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2020.2.254+0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "8d9496b2339095901106961f44718920732616bb"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.22"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OSQP]]
deps = ["BinaryProvider", "Libdl", "LinearAlgebra", "MathOptInterface", "MathProgBase", "OSQP_jll", "SparseArrays"]
git-tree-sha1 = "1231412a185b8c160111bdfa0c186108e2b5c049"
uuid = "ab2f91bb-94b4-55e3-9ba0-7f65df51de79"
version = "0.6.1"

[[OSQP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d0f73698c33e04e557980a06d75c2d82e3f0eb49"
uuid = "9c4f68bf-6205-5545-a508-2878b064d984"
version = "0.600.200+0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "7e4920a7d4323b8ffc3db184580598450bde8a8e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.7"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "ae6145ca68947569058866e443df69587acc1806"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.32"

[[Polynomials]]
deps = ["Intervals", "LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "0b15f3597b01eb76764dd03c3c23d6679a3c32c8"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProximalAlgorithms]]
deps = ["LinearAlgebra", "Printf", "ProximalOperators"]
git-tree-sha1 = "55476be8224b3db55f0bf862f2f84ac8597aaae5"
uuid = "140ffc9f-1907-541a-a177-7475e0a401e9"
version = "0.4.1"

[[ProximalOperators]]
deps = ["IterativeSolvers", "LinearAlgebra", "OSQP", "SparseArrays", "SuiteSparse", "TSVD"]
git-tree-sha1 = "d32ba50ff3e3192a3d4b9510618b9670d6925009"
uuid = "a725b495-10eb-56fe-b38b-717eba820537"
version = "0.14.2"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[QuartzImageIO]]
deps = ["FileIO", "ImageCore", "Libdl"]
git-tree-sha1 = "16de3b880ffdfbc8fc6707383c00a2e076bb0221"
uuid = "dca85d43-d64c-5e67-8c65-017450d5d020"
version = "0.7.4"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

[[RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5144e1eafb2ecc75765888a4bdcd3a30a6a08b14"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.24.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "b4912cd034cdf968e06ca5f943bb54b17b97793a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2884859916598f974858ff01df7dfc6c708dd895"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.3"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "ceeef74797d961aee825aabf71446d6aba898acb"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.2"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[StructuredOptimization]]
deps = ["AbstractOperators", "DSP", "FFTW", "LinearAlgebra", "ProximalAlgorithms", "ProximalOperators", "RecursiveArrayTools"]
git-tree-sha1 = "35bc5ae5ba1177ee119fabf40980d799c06b4ed2"
uuid = "46cd3e9d-64ff-517d-a929-236bc1a1fc9d"
version = "0.3.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TSVD]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "61cd1ce64b4ffb69e2d156ff7166a8eb796d699a"
uuid = "9449cd9e-2762-5aa3-a617-5413e99d722e"
version = "0.4.3"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "f91d170645a8ba6fbaa3ac2879eca5da3d92a31a"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.6.2"

[[TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "0f1017f68dc25f1a0cb99f4988f78fe4f2e7955f"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.7.1"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═a5d4303a-81b0-11ec-1083-ed13a1d29807
# ╟─f4cb8cfd-b6b1-4e93-9985-2c71b384093f
# ╠═62a8366f-14dc-410b-ba31-a8f695dfe986
# ╠═97006203-2819-489d-a4dc-cef3af6e72b3
# ╠═7415fc09-1155-49e5-b523-c9df2bd8d7d7
# ╠═bc36eb16-5b31-4639-864c-4ee4b627108e
# ╟─56a83328-2689-42be-8dfd-8ee38c5a70e0
# ╟─6e37dfac-5f50-498b-b356-f56fcbd7d02a
# ╟─68a5c2c4-e1ca-420d-b421-c4f570308884
# ╟─39f25941-cab4-4776-9167-5b75d555fdbf
# ╟─c899003d-bee5-4857-919a-8007d22c9f90
# ╟─135c1b29-8e82-4d06-b2f6-ff6fa7e3f031
# ╟─832052d7-c34d-4bce-8236-64cdc8f32be5
# ╟─df23221a-1dc6-4f45-9f03-35bc12c48e46
# ╠═7a9864b2-fc6f-4b89-8efa-183d644f8943
# ╠═d32b08c7-8262-4231-9469-df9f667e5c59
# ╟─c490029b-2b61-42c4-bd5f-72ba844e03b0
# ╟─7a104d94-6062-47a2-82cb-553513e28a86
# ╟─2bef7c97-8b88-4203-8825-eda70d5fab56
# ╠═9c19aaf2-3c25-425b-9e48-4d90d1366ce2
# ╠═4d4ec391-7219-459a-9b49-6ba901eeae92
# ╟─57a799e2-68b2-4867-867a-24d8590a84a5
# ╟─7c3a2e4b-419f-44f1-bea7-20c7a3aa06a3
# ╟─84e6d7fb-4df6-498b-bd5c-151004756e9e
# ╟─afb35a7d-1454-4c8f-a230-daabafc8e7a3
# ╟─ef90c37f-0e7e-48d1-8ab3-4ae779c3ed91
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
