import sys
import random 
import math
import numpy as np
f1=open('cleaned.data','r')
r=[]
for line in f1:
		line=line.strip()
		line=line.strip("'")
		line=line.split(',')
		r.append(line)

#print(r)

def convert(all_rows):
	for row in all_rows:
		for j in range(0,11):
			row[j]=int(row[j])

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

def euclidean_distance(row1,row2):
	distance=0
	for i in range(1,len(row1)-1):
		distance+=(row2[i]-row1[i])**2
	return math.sqrt(distance)

#print(distance(r[0],r[1]))

def nearest_neighbours(all_rows,ex_row,k):
	convert(all_rows)
	nn=[]
	for row in all_rows:
		dist=euclidean_distance(row,ex_row)
		n=[dist,row]
		nn.append(n)

	
	nn=sorted(nn)
	neighbours=[]
	for i in range(k):
		val=[nn[i][0],nn[i][1]]
		neighbours.append(val)
	return neighbours
	
f2=0
f4=0
def classify(nn,k):
	res=np.zeros(k,dtype=np.float32)
	s=0
	for i in range(len(nn)):
		if(nn[i][0]==0):
			s+=0
		else:
			res[i]+=1.0/nn[i][0]
			s+=res[i]
	for i in range(len(res)):
		res[i]/=s
		if res[i]=='nan':
			res[i]=0
		nn[i].append(res[i])
	cl=[]
	for i in range(len(nn)):
		x=nn[i][1][10]*nn[i][2]
		cl.append([x,nn[i][1][10]])
	fcl=[]
	x=0
	y=0	
	for i in range(len(cl)):
		if cl[i][1]==2:
			x=x+cl[i][0]
		elif cl[i][1]==4:
			y=y+cl[i][0]
	fcl.append([x,2])
	fcl.append([y,4])
	if fcl[0][0]>fcl[1][0]:
		return 2
	else:
		return 4
	#res=sorted(res)[:k]	
	#k_neighbours=nearest_neighbours(all_rows,ex_row,k)
	#nn=sorted(nn)[:k]

	#print(nn)
	'''
	freq2=0
	freq4=0
	for i in nn:
		#print(i)
		if i[1][10]==2:
			if i[0]==0:
				pass
			else:
				freq2=freq2+(1/i[0])
		elif i[1][10]==4:
			if i[0]==0:
				pass
			else:
				freq4=freq4+(1/i[0])
	#print(freq2,freq4)
	f2=freq2
	f4=freq4
	if freq2>freq4:
		return 2
	else:
		return 4
	'''
	'''
	labels=[]
	for row in neighbours:
		labels.append(row[1][-1])
	#labels1=set(labels)
	predicted_val=max(labels,key=labels.count)
	return predicted_val
	'''

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
			
			if(j==10):
				pass
			else:
				l.append(i[j])
		testset.append(l)
	return trainset,testset

def final_class(all_rows,ex_row,k):
	convert(all_rows)
	neighbours=nearest_neighbours(all_rows,ex_row,k)
	new_label=classify(neighbours,k)
	return new_label



sp=k_fold_validation(r,6)
#print(sp[0])
tr,te=make_splits(sp)
#max for k=4 then 3 ,this is only for normal knn


for i in range(len(te)):
	newcl=final_class(tr,te[i],100)
	#print(newcl)
	te[i].append(int(newcl))
#print(f2,f4)


acc=find_accuracy(tr,te)
acc=acc*100
print(acc)
"""
print(te)
"""
