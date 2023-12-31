import sys
import random 
import math

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
"""
convert(r)
for i in range(len(r)):
	print(r[i])
"""
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
"""					
trainingset=[]
testset=[]
for i in range(len(r)-1):
	
	for j in range(10):
		r[i][j]=int(r[i][j])
	
	if random.random()<0.75:
		trainingset.append(r[i])
	else:
		testset.append(r[i])
#print(len(trainingset))
"""
"""
for i in range(len(trainingset)):
	print(trainingset[i])
"""

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
"""
ex_row=[566800, 4, 1, 1, 2, 2, 1, 2, 3, 2]
convert(r)
ans=nearest_neighbours(r,ex_row,5)
print(ans)
"""

def classify(neighbours):
	#k_neighbours=nearest_neighbours(all_rows,ex_row,k)
	labels=[]
	for row in neighbours:
		labels.append(row[1][-1])
	#labels1=set(labels)
	predicted_val=max(labels,key=labels.count)
	return predicted_val

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
	new_label=classify(neighbours)
	return new_label
"""	
ex_row=[566800,10,5,5,3,6,7,7,10,1]
ans=nearest_neighbours(r,ex_row,5)
print(ans)
pred_label=final_class(r,ex_row,5)
print(pred_label)
"""
sp=k_fold_validation(r,6)
#print(sp[0])
tr,te=make_splits(sp)
#max for k=4 then 3
for i in range(len(te)):
	newcl=final_class(tr,te[i],100)
	te[i].append(int(newcl))
acc=find_accuracy(tr,te)
acc=acc*100
print(acc)
"""
print(te)
"""
