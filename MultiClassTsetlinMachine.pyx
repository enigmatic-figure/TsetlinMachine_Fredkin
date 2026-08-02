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

# This code implements a multiclass version of the Tsetlin Machine from paper arXiv:1804.01508
# https://arxiv.org/abs/1804.01508

#cython: boundscheck=False, cdivision=True, initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as np
import random
from libc.stdlib cimport rand, RAND_MAX

########################################
### The Multiclass Tsetlin Machine #####
########################################

cdef class MultiClassTsetlinMachine:
	cdef int number_of_classes
	cdef int number_of_clauses
	cdef int number_of_features
	cdef float s
	cdef int number_of_states

	cdef int[:,:,:] ta_state

	cdef int[:] clause_count
	cdef int[:,:,:] clause_sign

	cdef int[:] clause_output

	cdef int[:] class_sum

	cdef int[:] feedback_to_clauses

	cdef int threshold

	cdef int boost_true_positive_feedback
	cdef int clauses_per_class
	
	# Initialization of the Tsetlin Machine
	def __init__(self, number_of_classes, number_of_clauses, number_of_features, number_of_states, s, threshold, boost_true_positive_feedback = 0):
		cdef int i, j

		if number_of_classes <= 0:
			raise ValueError("number_of_classes must be > 0")
		if number_of_clauses <= 0:
			raise ValueError("number_of_clauses must be > 0")
		if number_of_features <= 0:
			raise ValueError("number_of_features must be > 0")
		if number_of_states <= 0:
			raise ValueError("number_of_states must be > 0")
		if threshold <= 0:
			raise ValueError("threshold must be > 0")
		if number_of_clauses % number_of_classes != 0:
			raise ValueError("number_of_clauses must be divisible by number_of_classes")

		self.number_of_classes = number_of_classes
		self.number_of_clauses = number_of_clauses
		self.number_of_features = number_of_features
		self.number_of_states = number_of_states
		self.s = s
		self.threshold = threshold
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.clauses_per_class = self.number_of_clauses // self.number_of_classes

		# The state of each Tsetlin Automaton is stored here.
		self.ta_state = np.random.randint(
			low=self.number_of_states,
			high=self.number_of_states + 2,
			size=(self.number_of_clauses, self.number_of_features, 2),
			dtype=np.int32
		)

		# Data structures for keeping track of which clause refers to which class, and the sign of the clause
		self.clause_count = np.zeros((self.number_of_classes,), dtype=np.int32)
		self.clause_sign = np.zeros((self.number_of_classes, self.clauses_per_class, 2), dtype=np.int32)
		
		# Data structures for intermediate calculations
		self.clause_output = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)
		self.class_sum = np.zeros(shape=(self.number_of_classes,), dtype=np.int32)
		self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)

		# Set up the Tsetlin Machine structure
		for i in range(self.number_of_classes):
			for j in range(self.clauses_per_class):
				self.clause_sign[i, self.clause_count[i], 0] = i * self.clauses_per_class + j
				if j % 2 == 0:
					self.clause_sign[i, self.clause_count[i], 1] = 1
				else:
					self.clause_sign[i, self.clause_count[i], 1] = -1

				self.clause_count[i] += 1

	cdef void calculate_clause_output(self, int[:] X, int predict=0):
		cdef int j, k
		cdef int action_include, action_include_negated
		cdef int all_exclude

		for j in range(self.number_of_clauses):				
			self.clause_output[j] = 1
			all_exclude = 1
			for k in range(self.number_of_features):
				action_include = self.action(self.ta_state[j,k,0])
				action_include_negated = self.action(self.ta_state[j,k,1])

				if action_include == 1 or action_include_negated == 1:
					all_exclude = 0

				if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
					self.clause_output[j] = 0
					break

			if predict == 1 and all_exclude == 1:
				self.clause_output[j] = 0

	cdef void sum_up_class_votes(self):
		cdef int target_class
		cdef int j

		for target_class in range(self.number_of_classes):
			self.class_sum[target_class] = 0

			for j in range(self.clause_count[target_class]):
				self.class_sum[target_class] += self.clause_output[self.clause_sign[target_class,j,0]] * self.clause_sign[target_class,j,1]
			
			if self.class_sum[target_class] > self.threshold:
				self.class_sum[target_class] = self.threshold
			elif self.class_sum[target_class] < -self.threshold:
				self.class_sum[target_class] = -self.threshold

	def predict(self, int[:] X):
		cdef int target_class
		cdef int max_class
		cdef float max_class_sum
		
		self.calculate_clause_output(X, predict=1)
		self.sum_up_class_votes()

		max_class_sum = self.class_sum[0]
		max_class = 0
		for target_class in range(1, self.number_of_classes):				
			if max_class_sum < self.class_sum[target_class]:
				max_class_sum = self.class_sum[target_class]
				max_class = target_class
			
		return max_class

	cdef int action(self, int state):
		if state <= self.number_of_states:
			return 0
		else:
			return 1

	def get_state(self, int clause, int feature, int automaton_type):
		return self.ta_state[clause,feature,automaton_type]

	def evaluate(self, int[:,:] X, int[:] y, int number_of_examples):	
		cdef int l, j
		cdef int errors
		cdef int max_class
		cdef float max_class_sum

		errors = 0
		for l in range(number_of_examples):
			self.calculate_clause_output(X[l], predict=1)
			self.sum_up_class_votes()

			max_class_sum = self.class_sum[0]
			max_class = 0
			for target_class in range(1, self.number_of_classes):				
				if max_class_sum < self.class_sum[target_class]:
					max_class_sum = self.class_sum[target_class]
					max_class = target_class
			
			if max_class != y[l]:
				errors += 1
		
		return 1.0 - 1.0 * errors / number_of_examples

	cpdef void update(self, int[:] X, int target_class):
		cdef int i, j, k
		cdef int negative_target_class
		cdef int action_include, action_include_negated
		cdef float v_target, p_target, v_neg, p_neg
		cdef int min_state = 1
		cdef int max_state = 2 * self.number_of_states

		if self.number_of_classes <= 1:
			negative_target_class = -1
		else:
			negative_target_class = rand() % self.number_of_classes
			while negative_target_class == target_class:
				negative_target_class = rand() % self.number_of_classes

		self.calculate_clause_output(X)
		self.sum_up_class_votes()

		for j in range(self.number_of_clauses):
			self.feedback_to_clauses[j] = 0

		# Target class feedback probability (clipped)
		v_target = <float>self.class_sum[target_class]
		if v_target > self.threshold:
			v_target = <float>self.threshold
		elif v_target < -self.threshold:
			v_target = <float>-self.threshold
		p_target = (self.threshold - v_target) / (2.0 * self.threshold)

		for j in range(self.clause_count[target_class]):
			if 1.0 * rand() / RAND_MAX > p_target:
				continue

			if self.clause_sign[target_class,j,1] >= 0:
				self.feedback_to_clauses[self.clause_sign[target_class,j,0]] = 1
			else:
				self.feedback_to_clauses[self.clause_sign[target_class,j,0]] = -1

		if negative_target_class >= 0:
			v_neg = <float>self.class_sum[negative_target_class]
			if v_neg > self.threshold:
				v_neg = <float>self.threshold
			elif v_neg < -self.threshold:
				v_neg = <float>-self.threshold
			p_neg = (self.threshold + v_neg) / (2.0 * self.threshold)

			for j in range(self.clause_count[negative_target_class]):
				if 1.0 * rand() / RAND_MAX > p_neg:
					continue

				if self.clause_sign[negative_target_class,j,1] >= 0:
					self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] = -1
				else:
					self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] = 1

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
							if self.boost_true_positive_feedback == 1 or 1.0 * rand() / RAND_MAX <= (self.s - 1.0) / self.s:
								if self.ta_state[j,k,0] < max_state:
									self.ta_state[j,k,0] += 1

							if 1.0 * rand() / RAND_MAX <= 1.0 / self.s:
								if self.ta_state[j,k,1] > min_state:
									self.ta_state[j,k,1] -= 1

						elif X[k] == 0:
							if self.boost_true_positive_feedback == 1 or 1.0 * rand() / RAND_MAX <= (self.s - 1.0) / self.s:
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
		cdef int target_class
		cdef long[:] random_index

		random_index = np.arange(number_of_examples)

		for epoch in range(epochs):			
			np.random.shuffle(random_index)

			for i in range(number_of_examples):
				example_id = random_index[i]
				target_class = y[example_id]
				self.update(X[example_id], target_class)
		return
