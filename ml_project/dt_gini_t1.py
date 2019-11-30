import sys
import random 
import math
import numpy as np
from math import exp

#filename=sys.stdin
filename=open('cleaned.data','r')
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

def convert(all_rows):
	for row in all_rows:
		for j in range(0,10):
			row[j]=int(row[j])

def k_fold_validation(all_rows,k):
	#convert(all_rows)
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

def normalise(x,mm):
	for r in x:
		#print(r)
		for i in range(len(r)):
			#print(i)
			r[i]=(r[i]-mm[i-1][0])/(mm[i-1][1]-mm[i-1][0])
#calculate min-max for each column
def maxmin(data):
	mm=[]
	for i in range(0,len(data[1])):
		c=[]
		for j in range(len(data)):
			for i in range(len(data[j])):
			#print(data[j][i])
				c.append(data[j][i])
				#print(c)
		maxx=max(c)
		minn=min(c)
		mm.append([maxx,minn])
	return mm





#sp=k_fold_validation(rows,2)
#tr,te=make_splits(sp)
#print(te)


def predict(row, coefficients):
	ypred = coefficients[0]
	for i in range(len(row)-1):
		ypred =ypred+ coefficients[i + 1] * row[i]
	#print(yhat)
	#print(1.0 / (1.0 + exp(-yhat)))
	return 1.0 / (1.0 + exp(-ypred))


#coeff=[-0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1


def stochasticgd(tr,epochs,learn_rate):
	'''coeff=[]
	for i in range(len(tr[0])): #check this
		coeff[i]=0	'''
	coeff = [0 for i in range(len(tr[0]))]
	#print(coeff)
	
	for i in range(epochs):
		summe=0
		for j in tr:
			yh=coeff[0]
			for ii in range(len(j)-1):
				yh=yh+(coeff[ii+1]*j[ii])
			ypred=1/(1+exp(-yh)) #sigmoid
			#print(yhat)'''
			#ypred=predict(j,coeff)
			error=round(j[-1],1)-round(ypred,1)
			#print(error)
			
			ferr=error**2
			summe=summe+ferr
			#print(summe)
			
			coeff[0]=coeff[0]+error*learn_rate*ypred*(1-ypred)
			for m in range(len(j)-1):
				coeff[m+1]=coeff[m+1]+error*learn_rate*ypred*(1-ypred)*j[m]
				#print(coeff)
		#print('>epoch=%d, lrate=%.5f, error=%.5f' % (i, learn_rate, summe))
	return coeff



def logistic_regress(train, test,epoch,learn_rate):
	prediction = []
	coeff =stochasticgd(train,epoch,learn_rate)
	cnt=0
	for row in test:
		#cnt+=1
		'''yh=coeff[0]
		for ii in range(len(row)-1):
			yh=yh+(coeff[ii+1]*row[ii])
		ypred=1/(1+exp(-yh)) #sigmoid
			#print(yhat)'''
		ypred = predict(row, coeff)
		ypred = round(ypred,1)
		prediction.append(ypred)
	#print(predictions)
	#print(cnt)
	return(prediction)
 

def find_accuracy(observed,predicted):
	correctly_classified=0
	for i in range(0,len(observed)):
		if(observed[i]==predicted[i]):
			correctly_classified=correctly_classified+1
	accuracy=correctly_classified/len(observed)
	return accuracy
			
sp=k_fold_validation(rows,6)
#print(sp[0])
tr,te=make_splits(sp)
minmax=maxmin(tr)
#print(minmax)
normalise(tr,minmax)
normalise(te,minmax)
#print(te)
#print(rows)
#x=stochasticgd(rows,1,0.1)
'''
#0.465,0.348
#0.4567,10000
0.45789,10000
0.45829,10000
0.73,1000
0.83,1000
0.93,1000
0.93,2000 -82.32758620689656
0.93,3000 --83.33333333333334
0.93089,5000--82.75862068965517

'''
pr=logistic_regress(tr,te,3000,0.93)
#print(pr)
actual=[(round(row[-1],1)) for row in tr]
acc=find_accuracy(actual,pr)
acc=acc*100

#print(tr)
#print(te)
#print(pr)
print(acc)
'''
for i in range(len(te)):
	newcl=final_class(tr,te[i],300)
	#print(newcl)
	te[i].append(int(newcl))
#print(f2,f4)


acc=find_accuracy(tr,te)
acc=acc*100
print(acc)
"""
print(te)
"""
'''

