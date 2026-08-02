# Copyright (c) 2025 Ole-Christoffer Granmo and University of Agder

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements a single-class version of the Tsetlin Machine from paper arXiv:1804.01508
# https://arxiv.org/abs/1804.01508

#cython: boundscheck=False, cdivision=True, initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as np
import random
from libc.stdlib cimport rand, RAND_MAX

#############################
### The Tsetlin Machine #####
#############################

cdef class TsetlinMachine:
	cdef int number_of_clauses
	cdef int number_of_features
	
	cdef float s
	cdef int number_of_states
	cdef int threshold

	cdef int[:,:,:] ta_state
	
	cdef int[:] clause_sign
	cdef int[:] clause_output
	cdef int[:] feedback_to_clauses

	# Initialization of the Tsetlin Machine
	def __init__(self, number_of_clauses, number_of_features, number_of_states, s, threshold):
		cdef int j

		self.number_of_clauses = number_of_clauses
		self.number_of_features = number_of_features
		self.number_of_states = number_of_states
		self.s = s
		self.threshold = threshold

		self.ta_state = np.random.randint(
			low=self.number_of_states,
			high=self.number_of_states + 2,
			size=(self.number_of_clauses, self.number_of_features, 2),
			dtype=np.int32
		)

		self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
		self.clause_output = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)
		self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)

		for j in range(self.number_of_clauses):
			if j % 2 == 0:
				self.clause_sign[j] = 1
			else:
				self.clause_sign[j] = -1

	cdef void calculate_clause_output(self, int[:] X):
		cdef int j, k
		cdef int action_include, action_include_negated

		for j in range(self.number_of_clauses):				
			self.clause_output[j] = 1
			for k in range(self.number_of_features):
				action_include = self.action(self.ta_state[j,k,0])
				action_include_negated = self.action(self.ta_state[j,k,1])

				if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
					self.clause_output[j] = 0
					break

	cpdef int predict(self, int[:] X):
		cdef int output_sum
		self.calculate_clause_output(X)
		output_sum = self.sum_up_clause_votes()

		if output_sum >= 0:
			return 1
		else:
			return 0

	cdef int action(self, int state):
		if state <= self.number_of_states:
			return 0
		else:
			return 1

	def get_state(self, int clause, int feature, int automaton_type):
		return self.ta_state[clause,feature,automaton_type]

	cdef int sum_up_clause_votes(self):
		cdef int output_sum
		cdef int j

		output_sum = 0
		for j in range(self.number_of_clauses):
			output_sum += self.clause_output[j] * self.clause_sign[j]
		
		if output_sum > self.threshold:
			output_sum = self.threshold
		elif output_sum < -self.threshold:
			output_sum = -self.threshold

		return output_sum

	def evaluate(self, int[:,:] X, int[:] y, int number_of_examples):
		cdef int l
		cdef int errors
		cdef int output_sum

		errors = 0
		for l in range(number_of_examples):
			self.calculate_clause_output(X[l])
			output_sum = self.sum_up_clause_votes()
			
			if output_sum >= 0 and y[l] == 0:
				errors += 1
			elif output_sum < 0 and y[l] == 1:
				errors += 1

		return 1.0 - 1.0 * errors / number_of_examples

	cpdef void update(self, int[:] X, int y):
		cdef int i, j, k
		cdef int action_include, action_include_negated
		cdef int output_sum
		cdef float prob
		cdef int min_state = 1
		cdef int max_state = 2 * self.number_of_states

		self.calculate_clause_output(X)
		output_sum = self.sum_up_clause_votes()

		for j in range(self.number_of_clauses):
			self.feedback_to_clauses[j] = 0
			
		if y == 1:
			prob = (self.threshold - output_sum) / (2.0 * self.threshold)
			for j in range(self.number_of_clauses):
				if 1.0 * rand() / RAND_MAX > prob:
					continue

				if self.clause_sign[j] >= 0:
					self.feedback_to_clauses[j] = 1
				else:
					self.feedback_to_clauses[j] = -1

		elif y == 0:
			prob = (self.threshold + output_sum) / (2.0 * self.threshold)
			for j in range(self.number_of_clauses):
				if 1.0 * rand() / RAND_MAX > prob:
					continue

				if self.clause_sign[j] >= 0:
					self.feedback_to_clauses[j] = -1
				else:
					self.feedback_to_clauses[j] = 1
	
		for j in range(self.number_of_clauses):
			if self.feedback_to_clauses[j] > 0:
				if self.clause_output[j] == 0:		
					for k in range(self.number_of_features):	
						if 1.0 * rand() / RAND_MAX <= 1.0 / self.s:								
							if self.ta_state[j,k,0] > min_state:
								self.ta_state[j,k,0] -= 1
													
						if 1.0 * rand() / RAND_MAX <= 1.0 / self.s:
							if self.ta_state[j,k,1] > min_state:
								self.ta_state[j,k,1] -= 1

				elif self.clause_output[j] == 1:					
					for k in range(self.number_of_features):
						if X[k] == 1:
							if 1.0 * rand() / RAND_MAX <= (self.s - 1.0) / self.s:
								if self.ta_state[j,k,0] < max_state:
									self.ta_state[j,k,0] += 1

							if 1.0 * rand() / RAND_MAX <= 1.0 / self.s:
								if self.ta_state[j,k,1] > min_state:
									self.ta_state[j,k,1] -= 1

						elif X[k] == 0:
							if 1.0 * rand() / RAND_MAX <= (self.s - 1.0) / self.s:
								if self.ta_state[j,k,1] < max_state:
									self.ta_state[j,k,1] += 1

							if 1.0 * rand() / RAND_MAX <= 1.0 / self.s:
								if self.ta_state[j,k,0] > min_state:
									self.ta_state[j,k,0] -= 1
					
			elif self.feedback_to_clauses[j] < 0:
				if self.clause_output[j] == 1:
					for k in range(self.number_of_features):
						action_include = self.action(self.ta_state[j,k,0])
						action_include_negated = self.action(self.ta_state[j,k,1])

						if X[k] == 0:
							if action_include == 0 and self.ta_state[j,k,0] < max_state:
								self.ta_state[j,k,0] += 1
						elif X[k] == 1:
							if action_include_negated == 0 and self.ta_state[j,k,1] < max_state:
								self.ta_state[j,k,1] += 1

	def fit(self, int[:,:] X, int[:] y, int number_of_examples, int epochs=100):
		cdef int i, epoch
		cdef int example_id
		cdef long[:] random_index

		random_index = np.arange(number_of_examples)

		for epoch in range(epochs):	
			np.random.shuffle(random_index)

			for i in range(number_of_examples):
				example_id = random_index[i]
				self.update(X[example_id], y[example_id])
		return
