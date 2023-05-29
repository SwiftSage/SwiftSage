table = """
41.67
48.67
35.33
27.33
29.67
6
27
81
54.67
86.67
94.33
100
75
100
64
17.33
9.33
83.33
66.67
39.67
100
94.33
100
41
21.67
100
65
100
100
18.33
"""

table_list = table.splitlines()[1:]

print(table_list[0])
print(table_list[29])

id2id = """
1-1     0
1-2     22
1-3     9
1-4     1
2-1     29
2-2     20
2-3     21
3-1     25
3-2     26
3-3     27
3-4     28
4-1     6
4-2     7
4-3     8
4-4     5
5-1     11
5-2     10
6-1     2
6-2     3
6-3     4
7-1     17
7-2     19
7-3     18
8-1     12
8-2     13
9-1     14
9-2     15
9-3     16
10-1    23
10-2    24
"""
id_order = []
for line in id2id.splitlines():
    if '    ' not in line:
        continue 
    id_order.append(int(line.split("    ")[1]))
print(id_order)

print("-"*50)
for id in id_order:
    # print(table_list[id])
    prev_c = ''
    line = ""
    text = table_list[id]
    # for i, c in enumerate(text):
    #     c = c.replace("\t", " ")
    #     if c == "\t" and i < len(text) and text[i+1] == "\t":
    #         continue 
    #     line += c
    print(text.replace("\t", " "))
    
"""
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
"""