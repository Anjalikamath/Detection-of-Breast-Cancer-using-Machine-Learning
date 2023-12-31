import sys

files=sys.stdin
rows=[]
for line in files:
	line=line.strip()
	line=line.split(',')
	rows.append(line)
cnt2=0
cnt4=0
for i in range(len(rows)):
	if int(rows[i][10])==2:
		cnt2=cnt2+1

	else:
		cnt4=cnt4+1

print(cnt2,cnt4)
