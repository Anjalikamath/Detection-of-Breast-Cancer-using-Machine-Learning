import sys
import math
import numpy as np
import pandas as pd
import csv


filename=sys.stdin
rows=[]

for line in filename:
	line=line.strip()
	line=line.split(',')
	l=[]
	for i in range(len(line)):
		if int(i)==0:
			pass
		else:
			l.append(int(line[i]))
	rows.append(l)


#entropy for the entire dataset
def fullentropy(x):
	p_minus=[] #2
	p_plus=[]  #4
	for i in range(len(x)):
		if(x[i][-1]==2):
			p_minus.append(x[i])
		else:
			p_plus.append(x[i])

	len_minus=len(p_minus)
	len_plus=len(p_plus)
	p_minus=len_minus/len(x)
	p_plus=len_plus/len(x)
	if(p_minus==0 or p_plus==0):
		return 0
	else:
		entro= ((-p_minus*(math.log(p_minus,2)))+(-p_plus*(math.log(p_plus,2))))
		return entro





#get gain for individual columns
def gain1(x,y):
	
	en=[]
	for j in range(1,11):
		one=[]
		for i in range(len(x)):
			#print(x[i])
			if(x[i][0]==j):
				one.append(x[i])

		one_en=(len(one)/len(x))*fullentropy(one)
		en.append(one_en)
	
	final_entropy=[]
	fullent=fullentropy(y)
	for i in range(len(en)):
		val=fullent-en[i]
		final_entropy.append(val)
	
	return final_entropy



#get the column with the max entropy
def find_max_entropy(x):
	maxx=x[0]
	col=0
	for i in range(1,len(x)):
		if (x[i]>maxx):
			maxx=x[i]
			col=i
	return col,maxx
	


'''
y=[]
for i in range(len(rows)):
	y.append(rows[i])
'''
full_entropy=fullentropy(rows)
x=(gain1(rows,rows)) #gain for all the columns
c,m=find_max_entropy(x) #get column with max gain


class node(object):
	def __init__(self,value,left,right,leaf,thresh):
	self.root_value=value
	self.root_left=left
	self.root_right=right
	self.threshold=thresh
	self.leaf=leaf
	
	def is_leaf(self):
		return self.leaf
	
	def get_threshold(self):
		return self.threshold
	def root(self):
		return self.root_value
	def lchild(self):
		return self.root_left

	def rchild(self):
		return self.root_right
	def __repr__(self):
		return "(%r,%r,%r,%r)"%(self.root_value,self.root_left,self.root_right,self.threshold)


class DT(object):
	














