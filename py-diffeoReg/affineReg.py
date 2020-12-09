import param_matching
import numpy as np

class affineRegistration:
	def __init__(self):
		self.tryGradient = True
	def localSearch(self, gamma0, Tsize, midPoint, nb_iterAff, type_group, enerAff):
		gamma = gamma0.copy()
		N = gamma.shape[0]
		nbDir=0
		Directions = 0
		coeff = 0.5
		if type_group == param_matching.TRANSLATION:
			nbDir = N
			Directions = np.zeros((nbDir, N + 1, N + 1))
			for i in range(N):
				Directions[i,i,N] = Tsize[i]
		elif type_group == param_matching.ROTATION:
			nbDir = N + (N*(N-1))/2
			Directions = np.zeros((nbDir, N + 1, N + 1))
			k=0
			u = 1/np.sqrt(2)
			for i in range(N):
				for j in range(i+1, N):
					Directions[k, i, j] = u
					Directions[k, j, i] = -u
					Directions[k, i, N] = -u*midPoint[j]
					Directions[k, j, N] = u*midPoint[i]
					k += 1

			for i in range(N):
				Directions[k,i, N] = Tsize[i]/10
				k += 1
		elif type_group == param_matching.SIMILITUDE:
			nbDir = N + (N*(N-1))/2 + 1
			Directions = np.zeros((nbDir, N + 1, N + 1))
			k=0
			u = 1/np.sqrt(2)
			v = 1 / np.sqrt(N)
			for i in range(N):
				for j in range(i+1, N):
					Directions[k, i, j] = u
					Directions[k, j, i] = -u
					Directions[k, i, N] = -u*midPoint[j]
					Directions[k, j, N] = u*midPoint[i]
					k += 1
			for i in range(N):
				Directions[k, i, i] = v
				Directions[k, i, N] = -v * midPoint[i]
				k += 1

			for i in range(N):
				Directions[k,i, N] = Tsize[i]/10
				k += 1

		elif type_group == param_matching.GENERAL:
			nbDir = N + N*N
			Directions = np.zeros((nbDir, N + 1, N + 1))
			k = 0
			for i in range(N):
				for j in range(N):
					Directions[k, i, j] = 1
					Directions[k, i, N] = -midPoint[j]
					k += 1


			for i in range(N):
				Directions[k, i, N] = Tsize[i] / 10
				k += 1

		elif type_group == param_matching.SPECIAL:
			nbDir = N + N*N - 1
			Directions = np.zeros((nbDir, N + 1, N + 1))
			k = 0
			u= 1/np.sqrt(2.0)

			for i in range(N):
				for j in range(N):
					Directions[k, i, j] = 1
					Directions[k, i, N] = -midPoint[j]
					k += 1
			for i in range(1, N):
				Directions[k, 0, 0] = u
				Directions[k, i, i] = -u
				Directions[k, 0, N] = -u*midPoint[0]
				Directions[k, i, N] = u*midPoint[i]
				k += 1

			for i in range(N):
				Directions[k, i, N] = Tsize[i] / 10
				k += 1

		grad = np.zeros(nbDir)
		oldGrad = np.zeros(nbDir)
		xGamma = np.zeros(nbDir)
		xGammaBest = np.zeros(nbDir)
		diffxGamma = np.zeros(nbDir)
		diffGrad = np.zeros(nbDir)
		p = np.zeros(nbDir)
		energies = np.zeros(2*nbDir+1)
		en = enerAff(gamma)
		oldEn = en

		print(f"initial energy {en:.2f}")
		istep = 0
		stepSequence = [0.5]

		while stepSequence[istep] > 0.005:
			istep +=1
			stepSequence.append(stepSequence[istep-1]*coeff)

		while (stepSequence[istep]  < 1):
			istep += 1
			stepSequence.append(stepSequence[istep - 1] * coeff)

		while (stepSequence[istep] > 0.005):
			istep += 1
			stepSequence.append(stepSequence[istep - 1] * coeff)


		H = np.eye(nbDir)
		istep = 0

		while istep < len(stepSequence):
			gradientStep = 10
			step = stepSequence[istep]
			cont = True
			it = 0
			while cont and it < nb_iterAff:
				it += 1
				cont = False
				bestEn = en
				gammaOld = gamma.copy()
				xGammaOld = xGamma.copy()
				for i in range(nbDir):
					gamma = gammaOld + step * Directions[i,:,:]
					energies[2*i] = enerAff(gamma)
					if energies[2*i] < bestEn:
						bestEn = energies[2*i]
						xGammaBest = xGammaOld.copy()
						xGammaBest[i] += step
						if energies[2*i] < 0.9999 * en:
							cont = True
					gamma = gammaOld - step * Directions[i,:,:]
					energies[2*i+1] = enerAff(gamma)
					if energies[2*i+1] < bestEn:
						bestEn = energies[2*i+1]
						xGammaBest = xGammaOld.copy()
						xGammaBest[i] -= step
						if energies[2*i+1] < 0.9999 * en:
							cont = True
					gamma.copy(Directions[i])
					grad[i] = (energies[2*i] - energies[2*i+1])/(step*2)



				if cont:
					en = bestEn
					print(f"Affine step= {step: .4f}  Energy: {en : .4f}  ({oldEn: .4f}")
					oldEn = en
					gamma[...] = 0
					xGamma = xGammaBest.copy()
					gamma += (xGamma[:,None,None]*Directions).sum(axis=0)
				else:
					print(f"Affine step= {step: .4f}  Energy: {en : .4f}")
					istep +=1
					gamma = gammaOld.copy()

				if it >= nb_iterAff:
					istep += 1

			if self.tryGradient:
				cont = True
			it  = 0
			while cont and it<nb_iterAff:
				it += 1
				gammaOld = gamma.copy()
				xGammaOld = xGamma.copy()
				for i in range(nbDir):
					gamma = gammaOld + step * Directions[i, :, :]
					energies[2*i] = enerAff(gamma)
					gamma = gammaOld - step * Directions[i, :, :]
					energies[2*i+1] = enerAff(gamma)
					grad[i] = (energies[2*i] - energies[2*i+1])/(step*2)
				diffGrad = grad - oldGrad
				oldGrad = grad.copy()
				sqGrad = np.sqrt((grad**2).sum() + 1e-10)
				p = np.dot(H, grad)
				xGammaOld = xGamma.copy()
				gradMat = (p[:,None,None]*Directions).sum(axis=0)
				alpha = gradientStep/sqGrad
				gamma = gammaOld - alpha * gradMat
				energies[2*nbDir] = enerAff(gamma)
				gradp = (p*grad).sum()

				while energies[2*nbDir] > en - 0.0001*alpha *gradp and gradientStep > 0.0000000001:
					gradientStep /= 1.5
					alpha = gradientStep / sqGrad
					gamma = gammaOld - alpha * gradMat
					energies[2 * nbDir] = enerAff(gamma)

				gradientStep *= 1.5
				if energies[2*nbDir] < en:
					print(f"gradient descent energy: {energies[2*nbDir] : 0.4f}")
					en = energies[2*nbDir]
					xGamma -= p*alpha
					diffxGamma = xGamma - xGammaOld

					rho = 1/(diffxGamma*diffGrad).sum()
					Z = np.eye(nbDir) - rho * np.outer(diffxGamma, diffGrad)
					newH = np.dot(Z,H)
					H = np.dot(newH, Z.T) + rho * np.outer(diffxGamma, diffGrad)
				else:
					cont = False
		return gamma, en
