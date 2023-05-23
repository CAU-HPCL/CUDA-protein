fp = open("test.txt", "r")

for line in fp:
    print (">> ", line.split())
    list1 = line.split()
    print (">> ", list1[0], float(list1[1]), list1[2])
    
