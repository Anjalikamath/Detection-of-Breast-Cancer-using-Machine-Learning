import sys
import random 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from random import seed
from random import randrange
from random import sample

f1=open('cleaned.data','r')
r=[]

for line in f1:
		line=line.strip()
		line=line.strip("'")
		line=line.split(',')
		l=[]
		for i in range(len(line)):
			if i==0:
				pass
			else:
				l.append(line[i])
		r.append(l)
#print(r)
def convert(all_rows):
	for row in all_rows:
		for j in range(0,10):
			row[j]=int(row[j])

def linear_combination(weights,inputs):
	total=weights[-1]	#add the bias to the sum,assuming it is the last element(since weights ar random, it shouldn't matter
	for j in range(len(weights)-1):
		total+=(weights[j]*inputs[j])
	#print(total)
	return total
	
def relu_act_func(i):  #neuron transfer,activation function
	if(i<=0):
		return 0
	else:
		return i

def k_fold_validation(all_rows,k):
	convert(all_rows)
	split_list=[]
	dataset=all_rows
	split_size=len(all_rows)/k
	split_size=int(split_size)

	for i in range(k):
		split=[]
		while(len(split)<split_size):
			j=random.randrange(0,len(all_rows))
			row=all_rows[j]
			split.append(row)
			split_list.append(row)
			removed_row=dataset.pop(j)
		#split_list.append(split)
	return split_list

def find_accuracy(observed,predicted):
	correctly_classified=0
	for i in range(0,len(observed)):
		if(observed[i]==predicted[i]):
			correctly_classified=correctly_classified+1
	accuracy=correctly_classified/len(observed)
	return accuracy

def make_splits(split_list):
	trainset=[]
	testset=[]
	for i in split_list:
		trainset.append(i)
		
		#print(i)
		l=[]
		for j in range(len(i)):
			
			if(j==9):
				pass
			else:
				l.append(i[j])
		testset.append(l)
	return trainset,testset


def new_network(no_inputs,no_hidden_layers,no_outputs):
	new_net=[]
	for i in range(0,no_hidden_layers):
		hidden_layer={}
		weights=[]
		weights=sample(range(1,100),no_inputs+1)
		weights=normalize_val(weights)
		#hidden_layer.append(weights
		hidden_layer=[{'w':weights,'op':[],'err':0}]
	new_net.append(hidden_layer)

	for i in range(0,no_outputs):
		output_layer={}
		weights_o=[]
		weights_o=sample(range(1,100),no_hidden_layers+1)
		#print(weights_o)
		weights_o=normalize_val(weights_o)
		#print(weights_o)
		output_layer=[{'w':weights_o,'op':[],'err':0}]
		#print(output_layer)
	new_net.append(output_layer)
	#print(new_net)
	return new_net

def normalize_val(a):
	
	new_a=[]
	new_a=stats.zscore(a)
	ans=[]
	for i in range(len(new_a)):
		ans.append(abs(new_a[i]))
	"""
	for i in range(len(a)):
		new_val=(a[i]-min_value)/(max_value-min_value)
		new_a.append(new_val)
	"""

	return ans

def forward_propogation(nn,input_row):
	next_lr_ip=[]
	for j in range(len(nn)):
		#next_lr_ip=[]
		#print(nn[j])
		for i in nn[j]:
			#print(i['op'])
			#temp_list=[]
			total=linear_combination(i['w'],input_row)
			#print(total)
			#print(relu_act_func(total))
			#i['op']=relu_act_func(total)
			#print(relu_act_func(total))
			i['op'].append(relu_act_func(total))
			#temp_list.append(relu_act_func(total))
			print(i['op'])
			next_lr_ip.append(i['op'])
	#print(next_lr_ip)
	return next_lr_ip

def slope(output):
	return output*(1-output)

def back_propogation_error(network,observed):
	for i in reversed(range(len(network))):
		layer=network[i]
		layer_error=[]
		if(i!=(len(network)-1)):
			for x in range(len(layer)):
				error=0
				for j in network[i+1]:
					error+=(j['w'][x]*j['err'])
				layer_error.append(error)
		else:
			for k in range(len(layer)):
				unit=layer[k]
				layer_error.append(observed[k]-unit['op'])
		for j in range(len(layer)):
			n=layer[j]
			n['err']=layer_error[j]*slope(n['op'])
		
def weights_update(nn,ip_row,eta):
	for i in range(len(nn)):
		nxt_lr_ip=ip_row[:-1]  #do not update weight of bias
		if(i!=0):	#no weights for first layer i.e ip

			'''for n in nn[i-1]:
				output=nn['op']
				nxt_lr_ip.append(output)
		for n in nn[i]:
			for k in range(len(nxt_lr_ip)):
				n['w'][k]+=eta*n['err']*nxt_lr_ip[k]
			n['w'][-1]+=eta*n['err']'''
			nxt_lr_ip= [neuron['op'] for neuron in nn[i - 1]]
		for neuron in nn[i]:
			for j in range(len(nxt_lr_ip)):
				neuron['w'][j] += eta* neuron['err'] *nxt_lr_ip[j]
			neuron['w'][-1] += eta* neuron['err']	

def train_network(nn,tr,eta,no_epochs,no_outputs,actual_values):
	#print(nn)
	her=0
	for i in range(no_epochs):
		total_error=0
		
		for j in range(len(tr)):
			output=[]
			output=forward_propogation(nn,tr[j])
			#print(nn)
			#print(output)
			"""
			exp=[]
			for k in range(no_outputs):
				exp.append(0)
			#exp[j][-1]=1 #for bias
			"""
			exp=actual_values[j]
			error=[]
			"""
			for l in range(len(exp)):
				for j in range(len(output)):
					err=(exp[l]-output[j][0])**2
					error.append(err)
			"""
			error=(exp-output[1][0])**2
			#print(error)
			#nn['err']=error
			#print(nn)
			"""
			for m in range(len(error)):
				total_error+=error[m]
			"""
			her+=error
		nn[0]['err']=her
		back_propogation_error(nn,exp)
		weights_update(nn,j,eta)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (i,eta,error))
"""
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propogation(network, row)
			expected = [0 for i in range(n_outputs)]
			#expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i][0])**2 for i in range(len(expected))])
			back_propogation_error(network, expected)
			weights_update(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
"""
		
def predict(nn,ip_row):
	output=forward_propogation(nn,row)
	new_label=max(output)
	for i in range(len(output)):
		if(output[i]==new_label):
			return i		


seed(1) #seed is used to save the state of random function so that it can generate some random numbers on multiple execution of the code.
'''network = [[{'w': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'w': [0.2550690257394217, 0.49543508709194095]}, {'w': [0.4494910647887381, 0.651592972722763]}]]

row = [1, 0, None]
output = forward_propogation(network, row)
print(output)

for layer in network:
	print(layer)

'''
convert(r)
network=new_network(9,1,1)

#print(network)
"""
for layer in network:
	print(layer)
"""
dataset=k_fold_validation(r,6)
train,test=make_splits(dataset)
#print(train)


eta=0.1
actual_values=[]
for i in train:
	actual_values.append(i[9])
#print(actual_values)
"""
intermediate_op=[]
for i in range(len(train)):
	output_label=forward_propogation(network,train[i])
	intermediate_op.append(output_label)
"""
train_network(network,train,eta,500,1,actual_values)
predicted=[]
for i in range(len(test)):
	output=predict(network,test[i])
	predicted.append(output)
final_acc=find_accuracy(observed,predicted)
final_acc=final_acc*100
#print(final_acc)

'''
convert(r)
network=new_network(3,1,1)
#print(network)
row = [1, 0, 2]
output = forward_propogation(network, row)
#print(output)

for layer in network:
	for i in layer:
		print(i)
'''

