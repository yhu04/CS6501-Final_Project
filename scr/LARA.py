import ast
import csv 
import math
import numpy as np
from numpy.linalg import inv
from scipy import random, linalg
from scipy.optimize import minimize

def load_data(text_file,vocab_file):
	overall_rating = {}
	word_index = {}
	W = {}
	restaurant = []

	review_file = open(text_file).read().split('\n')[:-1]
	for line in review_file:
		temp = line.split(':')
		restaurant_id = temp[0]
		new_score = float(temp[1])
		new_index = np.array(ast.literal_eval(temp[2]))
		new_matrix = np.array(ast.literal_eval(temp[3]))
		if new_index.shape[0] == 0:
			continue

		restaurant.append(restaurant_id)

		if restaurant_id in overall_rating:
			overall_rating[restaurant_id].append(new_score)
		else:
			overall_rating[restaurant_id] = [new_score]

		if restaurant_id in word_index:
			word_index[restaurant_id].append(new_index)
		else:
			word_index[restaurant_id] = [new_index]

		if restaurant_id in W:
			W[restaurant_id].append(new_matrix)
		else:
			W[restaurant_id] = [new_matrix]
	
	vocab_file = open(vocab_file).read().split('\n')
	vocab = []
	for line in vocab_file:
		str_split = line.split(',')
		for term in str_split:
			vocab.append(term)
	return overall_rating, word_index, W, vocab, list(set(restaurant))

def initialize(aspect_number,word_number):
	mu = np.random.rand(aspect_number)
	sigma = np.eye(aspect_number)
	delta = 3
	beta = np.random.rand(aspect_number,word_number)
	return mu, sigma, delta, beta 

def Mu(alpha):
	return np.mean(alpha, axis=0)

def Sigma(alpha,mu):
	S = np.dot((alpha - mu).T, (alpha - mu)) / alpha.shape[0]
	w,v = np.linalg.eig(S)
	w[w<1e-3] = 0.0
	return np.dot(w * v, v.T)

def Beta(r,beta,W,alpha,delta,word_index):
	n_reviews = alpha.shape[0]
	k = alpha.shape[1]
	n_words = beta.shape[1]
	beta = beta.reshape((k * n_words))
	def f(beta, r, alpha, W, delta, word_index):
		p = np.exp(alpha)
		p = p / np.sum(p, axis=1, keepdims=True)
		sum = 0
		for i in range(n_reviews):
			sum += (r[i] - np.dot(p[i], np.sum(np.reshape(beta, (k, n_words))[:, word_index[i]] * W[i], axis=1))) ** 2 / (2 * delta)
		return sum

	def g(beta, r, alpha, W, delta, word_index):
		p = np.exp(alpha)
		p = p / np.sum(p, axis=1, keepdims=True)
		d_beta = np.zeros((k, n_words))
		for j in range(k):
			for i in range(n_reviews):
				d_beta[j][word_index[i]] += (np.dot(p[i], np.sum(np.reshape(beta, (k, n_words))[:, word_index[i]] * W[i], axis=1)) - r[i]) * p[i][j] * W[i][j]
		return d_beta.reshape((k * n_words)) / delta

	res = minimize(f, beta, args=(r, alpha, W, delta, word_index), jac=g, method='L-BFGS-B', options={'disp': False})
	return res.x.reshape((k, n_words))

def Delta(r,alpha,s):
	n_reviews = alpha.shape[0]
	p = np.exp(alpha)
	p = p / np.sum(p, axis=1, keepdims=True)
	return np.mean(np.square(np.array(r) - np.sum(p * s, axis = 1)))

def Negative_Log_likelihood(r,alpha,beta,W,delta,mu,sigma,word_index):
	n_reviews = alpha.shape[0]
	k = alpha.shape[1]
	p = np.exp(alpha)
	p = p / np.sum(p, axis=1, keepdims=True)

	# data likelihood
	data_likelihood = 0

	for i in range(n_reviews):
		data_likelihood += (r[i] - np.dot(p[i], np.sum(beta[:, word_index[i]] * W[i], axis=1)))**2 / (2 * delta)
	
	data_likelihood = data_likelihood / n_reviews + math.log(delta)

	# alpha likelihood
	alpha_likelihood = np.sum((alpha - mu) * np.dot(alpha - mu, inv(sigma + 1e-3 * np.eye(k)))) + math.log(np.linalg.det(sigma + 1e-3 * np.eye(k)))
	
	# beta likelihood
	beta_likelihood = np.sum(np.square(beta)) * 1e-3

	return data_likelihood + alpha_likelihood + beta_likelihood

def alpha_inference(alpha, r, s, delta, mu, inv_sigma):
	k = alpha.shape[0]
	def f(alpha, r, s, delta, mu, inv_sigma):
		p = np.exp(alpha)
		p = p / np.sum(p)
		return (r - np.dot(p, s)) ** 2 / (2 * delta) + 0.5 * np.dot(alpha - mu, np.dot(inv_sigma, alpha - mu))

	def g(alpha, r, s, delta, mu, inv_sigma):
		p = np.exp(alpha)
		p = p / np.sum(p)
		dpda = -np.outer(p, p)
		dpda[np.diag_indices_from(dpda)] += p
		return np.dot(np.dot((np.dot(p, s) - r), s), dpda) / delta + np.dot(inv_sigma, alpha - mu)

	res = minimize(f, alpha, args=(r, s, delta, mu, inv_sigma), jac=g, method='L-BFGS-B', options={'disp': False})
	return res.x

def e_step(r,mu,sigma,delta,beta,W,alpha,word_index):
	print "E Step:"
	n_reviews = alpha.shape[0]
	k = alpha.shape[1]
	new_alpha = np.zeros(alpha.shape)
	s = np.zeros((n_reviews, k))
	
	for i in range(n_reviews):
		s[i] = np.sum(beta[:, word_index[i]] * W[i], axis=1)
		new_alpha[i] = alpha_inference(alpha[i], r[i], s[i], delta, mu, inv(sigma + 1e-3 * np.eye(k)))
	return new_alpha, s

def m_step(alpha,r,s,W,beta,word_index):
	print "M step"
	n_reviews = len(r)
	mu = Mu(alpha)
	sigma = Sigma(alpha, mu)
	delta = Delta(r, alpha, s)
	beta = Beta(r,beta,W,alpha,delta,word_index)
	return mu,sigma,delta,beta

def run_em(maxIter,r,W,word_index,word_number):
	aspect_number = 5
	n_reviews = r.shape[0]

	mu,sigma,delta,beta = initialize(aspect_number,word_number)
	alpha = np.random.multivariate_normal(mu, sigma, n_reviews)

	iter = 0
	converge = False
	old_log = Negative_Log_likelihood(r,alpha,beta,W,delta,mu,sigma,word_index)
	while iter < maxIter and not converge:		
		alpha, s = e_step(r,mu,sigma,delta,beta,W,alpha,word_index)
		mu,sigma,delta,beta = m_step(alpha,r,s,W,beta,word_index)
		new_log = Negative_Log_likelihood(r,alpha,beta,W,delta,mu,sigma,word_index)

		print "Log_likelihood: " + str(new_log)
		difference = old_log - new_log

		print difference

		if abs(difference)<0.001:
			converge=True
			print("EM algorithm converges in "+str(iter+1)+" iterations")
			return s
		iter = iter + 1
		if iter == maxIter:
			print("EM algorithm fails to converge in "+str(iter)+" iterations")

		old_log = new_log
		print "r:"
		print r
		p = np.exp(alpha)
		p = p / np.sum(p, axis=1, keepdims=True)
		print "p:"
		print p
		print "pred:"
		print np.sum(p * s, axis=1)
	return s

def main():

	overall_rating, word_index, W, vocab, restaurant = load_data('review_data.txt','vocab.txt')
	print "Finish Load Data"

	aspect_rating = []
	for id_num in restaurant:
		restaurant_rating = np.array(overall_rating[id_num])
		restaurant_W = W[id_num]
		restaurant_word_index = word_index[id_num]
		n_reviews = restaurant_rating.shape[0]
		word_number = len(vocab)
		s = run_em(10,restaurant_rating,restaurant_W,restaurant_word_index,word_number)
		temp = [id_num]
		aspect_rating.append(','.join(temp + [str(x) for x in np.sum(s, axis=0)/n_reviews]))
		print "Finish restaurant"+id_num
		print aspect_rating[-1]

	print aspect_rating
	open("aspect_rating.csv",'w').write('\n'.join(aspect_rating))

if __name__ == '__main__':
	main()
