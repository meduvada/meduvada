------------Linear Seperation Easy-------------

import matplotlib.pyplot as plt
import numpy as np
print("practical  performed by ")
fig, ax=plt.subplots()
xmin,xmax = -0.2,1.4
ymin,ymax=-0.1,1.1
X=np.arange(xmin,xmax,0.1)
ax.scatter(0,0,color='r')
ax.scatter(0,1,color='r')
ax.scatter(1,0,color='r')
ax.scatter(1,1,color='g')
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
m,c=-1,1.2
ax.plot(X,m*X+c)
plt.show()

-----------------------------Linear Diff Hard----------------------

linear diff
import numpy as np
import matplotlib.pyplot as plt
def create_distance_function(a, b, c):

    def distance(x, y):
       
        nom = a * x + b * y + c
        if nom == 0:
            pos = 0
        elif (nom<0 and b<0) or (nom>0 and b>0):
            pos = -1
        else:
            pos = 1
        return (np.absolute(nom) / np.sqrt( a ** 2 + b ** 2), pos)
    return distance
orange = (4.5, 1.8)
lemon = (1.1, 3.9)
fruits_coords = [orange, lemon]
fig, ax = plt.subplots()
ax.set_xlabel("sweetness")
ax.set_xlabel("sweetness")
x_min, x_max = -1, 7
y_min, y_max = -1, 8
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
X = np.arange(x_min, x_max, 0.1)
step = 0.05
for x in np.arange(0, 1+step, step):
    slope = np.tan(np.arccos(x))
    dist4line1 = create_distance_function(slope, -1, 0)
    Y = slope * X
    results = []
    for point in fruits_coords:
        results.append(dist4line1(*point))
    if (results[0][1] != results[1][1]):
        ax.plot(X, Y, "g-", linewidth=0.8)
    else:
        ax.plot(X, Y, "r-", linewidth=0.8)
size = 10
for (index, (x, y)) in enumerate(fruits_coords):
    if index == 0:
        ax.plot(x, y, "o",
                color="darkorange",
                markersize=size)
    else:
        ax.plot(x, y, "oy",
                markersize=size)
plt.show()

--------------------------------Hop network---------------------------------

clear all;
function [yin,y]=net(j,w,x,temp,theta,yin,y)
    yin=yin
    y=y
    for i =1:4
        temp=temp+(y(i)*w(i,j));
    end
    yin(j)=x1(j)+temp;
if(yin(j)>theta)
    y(j)=1;
elseif(yin(j)==theta)
y(j)=yin(j);
else
    y(j)=-1
    end
    disp(j)
disp(y)
endfunction
disp("Discrete Hopfield network");
theta=0;
x=[1 1 1 -1]
w=x'*x
disp("weight matrix with self connection");
disp(w);
for i=1:4
    for j=1:4
        if(i==j)
            w(i,j)=0
            end
             end
            end
            disp("Weight matrix with no self connection")
disp(w);
disp("Given input pattern for testing");
x1=[1 1 1 -1]
y=[1 1 1 -1]
temp=0
disp("by asynchronous updation method");
disp("The net input calculated is: ");
yin=[0 0 0 0]
[yin,y]=net(1,w,temp,theta,yin,y)
[yin,y]=net(4,w,temp,theta,yin,y)
[yin,y]=net(3,w,temp,theta,yin,y)
[yin,y]=net(2,w,temp,theta,yin,y)
disp("The output calculated from net input is: ")
disp(y)

--------------------------------binary sigmoidal-------------------------------


import math

print("Practical  performed by ")
n = int(input("Enter no. of input neurons: "))

print("Enter input")
inputs = []
for i in range(0, n):
    x = float(input())
    inputs.append(x)
print(inputs)

print("Enter weight")
weights = []
for i in range(0, n):
    w = float(input())
    weights.append(w)
print(weights)

print("The net input can be calculated as Yin=x1w1+x2w2+x3w3")
Yin = []
for i in range(0, n):
    Yin.append(inputs[i] * weights[i])

ynet = round(sum(Yin), 3)
print("Net input for y neuron=", ynet)

print("Apply activation function over net input A) Binary Sigmoidal activation function")
y = round((1 / (1 + math.exp(-ynet))), 3)
print(y)


----------------------------------------delta rule---------------------------------------


import numpy as np
import time
print("Practical  Performed by ")
np.set_printoptions(precision=9)
x=np.zeros((3,))
weights=np.zeros((3,))
desired=np.zeros((3,))
actual=np.zeros((3,))
for i in range(0,3):
    x[i]=float(input("Initial Inputs:"))
for i in range(0,3):
    weights[i]=float(input("Initial Weights:"))
for i in range(0,3):
    desired[i]=float(input("Desired output:"))
for i in range(0,3):
    actual=x[i]*weights
print("actual",actual)
print("desired",desired)
a=float(input("Enter Learning Rate:"))
while True:
    if np.array_equal(desired,actual):
        break #no change
    else:
        for i in range(0,3):
            weights[i]=weights[i]+a*(desired[i]-actual[i])
            actual=x[i]*weights
        print("Weights",weights)
        print("Actual",actual)
        print("desired",desired)
print("FINAL OUTPUT")
print("Corrected Weights",weights)
print("Actual",actual)


----------------------------Fuzzy Ratio---------------------------------



print("practical performed by ")
import fuzzywuzzy
from fuzzywuzzy import fuzz
Str_A ='Fuzzywuzzy is a lifesaver!'
Str_B ='fuzzy wuzzy is a LIFE SAVER.'
ratio = fuzz.ratio(Str_A.lower(), Str_B.lower())
print('Similariy score: {}' .format(ratio))
Str_A ='Chicago Illinois'
Str_B ='chicago'
ratio = fuzz.partial_ratio(Str_A.lower(), Str_B.lower())
print('partial_ratio: {}' .format(ratio))
Str_A ='Gunner William kline'
Str_B ='Kline, Gunner William'
ratio = fuzz.token_sort_ratio(Str_A, Str_B)
print('token_sort_ratio: {}' .format(ratio))
Str_A ='The 300 meter steepliechase winner, Soufiane E1 Bakkali'
Str_B ='Soufiane E1 Bakkali'
ratio = fuzz.token_set_ratio(Str_A, Str_B)
print('token_set_ratio: {}' .format(ratio))



---------------------------------simple linear networks--------------------------

print('Pract performed by sahil kadam')
x = int(input("Enter the value for x "))
b = int(input("Enter the value for bias "))
w = int(input("Enter the value for weight "))

ynet = w * x + b

print("Net input = ", ynet)

print("Apply Activation Function over net input, Ram Function")

if ynet < 0:
    output = 0
elif ynet >= 0 and ynet <= 1:
    output = ynet
else:
    output = 1

print("output = ", output)



----------------------------------and/not mculloh---------------------------

print("Practical performed by ")
num_ip = int(input("Enter No of Inputs: "))
print("For the", num_ip, "input calculate the net input")

x1 = []
x2 = []
t = []

for i in range(0, num_ip):
    ele1 = int(input("x1="))
    ele2 = int(input("x2="))
    out = int(input("y="))
    x1.append(ele1)
    x2.append(ele2)
    t.append(out)

w1 = int(input("Enter weight value of input1: "))
w2 = int(input("Enter weight value of input2: "))
n = [w1 * i for i in x1]
m = [w2 * i for i in x2]
Yin = []

for i in range(0, num_ip):
    Yin.append(n[i] + m[i])
print("Net inputs")
print("Yin=", Yin)

Y = []
for i in range(0, num_ip):
    if (Yin[i] >= 1):
        ele = 1
        Y.append(ele)
    if (Yin[i] < 1):
        ele = 0
        Y.append(ele)

print("Y=", Y)
print("T=", t)

if Y == t:
    print("Weight values accepted.")
else:
    print("Weights are not suitable.")


---------------------------------hebb and-------------------------------


print("Practical  Performed by ")
X = [[1,1], [1,-1], [-1,1], [-1,-1]]
print("Inputs = ")
for x in X:
    print(x)
Y = [1,-1,-1,-1]
print("Target = ",Y)
w = [0,0]
print("Initial weight values = ", w)
for i in range(len(X)):
    for j in range(len(w)):
        w[j] = w[j] + X[i][j] * Y[i]
    print(i, "Iteration, Weight values = ",w)
	
	
------------------------------------backpropagation------------------------------

import math
import numpy as np
print("Practical  Performed by ")
np.set_printoptions(precision=4)
v1=np.array([0.6,-0.3])
v2=np.array([-0.1,0.4])
w=np.array([-0.2,0.4,0.1])
b=np.array([0.3,0.5])
x=np.array([0,1])
alpha=0.25
t=1
zin=[]
print("Calculate net input to z layer")
for i in range(0,2):
    zin.append(round(b[i]+x[0]*v1[i]+x[1]*v2[i],4))
print("Net Input for")
print("z1=",zin[0])
print("z2=",zin[1])
z=[]
print("Apply activation function to calculate output")
for i in range(0,2):
    z.append(round(1/(1+math.exp(-zin[i])),4))
print("output for")
print("z1=",z[0])
print("z2=",z[1])
print("calculate net input to output layer")
yin=w[0]+z[0]*w[1]+z[1]*w[2]
print("yin=",yin)
print("calculate net output")
y=round(1/(1+math.exp(-yin)),4)
print("yin=",y)
fyin=y*(1-y)
dk=round((t-y)*fyin,4)
print("dk",dk)
dw1=alpha*dk*z[0]
dw2=alpha*dk*z[1]
dw0=alpha*dk
print("compute error portion in delta")
din=[]
for i in range(1,3):
    din.append(dk*w[i])
print("din1=",din[0])
print("din2=",din[1])
print("Error in delta")
fzin=[]
d=[]
for i in range(0,2):
    fzin.append(round(z[i]*(1-z[i]),4))
    d.append(round(din[i]*fzin[i],4))
print("fzin1=",fzin[0])
print("fzin2=",fzin[1])
print("d1=",d[0])
print("d2=",d[1])
print("changes in weights between input and hidden layer")
dv=[[0,0],[0,0],[0,0]]
for i in range(0,3):
    for j  in range(0,2):
        if i<=1:
            dv[i][j]=alpha*d[j]*x[i]
        else:
            dv[i][j]=alpha*d[j]
print(dv)
print("Final weights of network")
for i in range(0,2):
    v1[i]=v1[i]+dv[0][i]
print("v1=",v1)
for i in range(0,2):
    v2[i]=v2[i]+dv[1][i]
print("v2=",v2)
for i in range(0,2):
    b[i]=b[i]+dv[2][i]
print("b1=",b[0])
print("b2=",b[1])
w[1]=w[1]+dw1
w[2]=w[2]+dw2
w[0]=w[0]+dw0
print("w=",w)


-----------------------------------------xor mculloh-------------------------------------

import numpy as np
print("Practical 2(B) Performed by ")
print("Enter Weights")
w11=int(input("weight w11="))
w12=int(input("weight w12="))
w21=int(input("weight w21="))
w22=int(input("weight w22="))
v1=int(input("weight v1="))
v2=int(input("weight v2="))
print("Enter threshhold values")
theta=int(input("theta="))
x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])
t=np.array([0,1,1,0])
z1=np.zeros((4,))
z2=np.zeros((4,))
y=np.zeros((4,))
zin1=np.zeros((4,))
zin2=np.zeros((4,))
zin1=(x1*w11)+(x2*w21)
zin2-(x1*w12)+(x2*w22)
print("z1 ",zin1)
print("z2 ",zin2)
for i in range(0,4):
    if zin1[i]>=theta:
        z1[i]=1
    else:
        z1[i]=0     
yin=np.array([])
yin=(z1*v1)+(z2*v2)
for i in range(0,4):
    if yin[i]>=theta:
        y[i]=1
    else:
        y[i]=0        
print("yin=",yin)
print("Output of net")
y=y.astype(int)
print("y",y)
print("t",t)

----------------------SCILAB----------------------

---------------------------Discrete Hopfield network------------------------

disp("Name ")
clear all;
function [yin, y]=net(j, w, x, temp, theta, yin, y)
 yin=yin
 y=y
 for i=1:4
 temp=temp+(y(i)*w(i,j));
 end
 yin(j)=x1(j)+temp;
if(yin(j)>theta)
 y(j)=1;
elseif(yin(j)==theta)
y(j)=yin(j);
else
 y(j)=-1
 end
 disp(yin)
disp(y)
endfunction
disp("Discrete Hopfield network");
theta=0;
x=[1 1 1 -1]
w=x'*x
disp("weight matrix with self connection");
disp(w);
for i=1:4
for j=1:4
 if(i==j)
 w(i,j)=0
 end
 end
 end
 disp("Weight Matrix with no self Connection");
disp(w);
disp("Given input pattern for testing");
x1=[1 1 1 -1]
y=[1 1 1 -1]
temp=0;
disp("By Asynchronus updation method");
disp("The net input calculated is:");
yin=[0 0 0 0]
[yin,y]=net(1,w,temp,theta,yin,y)
[yin,y]=net(4,w,temp,theta,yin,y)
[yin,y]=net(3,w,temp,theta,yin,y)
[yin,y]=net(2,w,temp,theta,yin,y)
disp("The output calculated from net input is:")
disp(y)

-------------------------------Kohonen self organizing--------------------------------

disp("Name :");
clear all;
disp('Kohonen self organizing feature maps');
disp('The input patterns are')
x=[0 0 1 1; 1 0 0 0; 0 1 1 0; 0 0 0 1]
alpha=0.5;
disp('Since we have 4 input pattern and cluster unit to be formed is 2, the weight matrix is');
w=[0.2 0.9;0.4 0.7; 0.6 0.5; 0.8 0.3]
disp('The Learning rate of this epoch is');
alpha
i=1;
j=1;
k=1;
m=1;
while(k<=4)
for j=1:2
 temp=0;
 for i=1:4
 temp=temp + ((w(i,j)-x(k,i))^ 2);
 end
 D(j)=temp
end
if(D(1)<D(2))
 J=1;
else
 J=2
end
disp('The winning unit is');
disp(J)
disp('Weight updation');
for m=1:4
 w(m,J)=w(m,J)+(alpha*(x(k,m)-w(m,J)));
 end
disp(w)
k=k+1;
disp(k)
end


-----------------------------Adaptive Resonance Theory Network---------------------------

disp("Name")
clear all;
disp('Adaptive Resonance Theory Network 1');
L=2;
m=3;
n=4;
rho=0.4;
te=L/(L-1+n);
te=te/2;
b=[te te te;te te te;te te te;te te te];
t=ones(3,4);
s=[1 1 0 0;0 0 0 1;1 0 0 0;0 0 1 1];
e=1
while(e<=4)
 temp=0;
 for i=1:4
 temp=temp+s(e,i);
 end
 ns=temp;
 x(e,:)=s(e,:);
 for i=1:3
 temp=0;
 for j=1:4
temp=temp+(x(e,j)*b(j,i));
 end
 yin(i)=temp;
 end
 j=1;
 if (yin(j)>=yin(j+1)& yin(j)>= yin(j+2)) then
 J=1;
 elseif (yin(j+1)>= yin(j) & yin(j+1)>=yin(j+2))
 J=2;
 else
 J=3;
 end
 disp("J=");
 disp(J);
 for i=1:4
 x1(i)=x(e,i)*t(J,i);
 end
 x1;
 temp=0;
 for i=1:4
 temp=temp+x1(i);
 end
 nx=temp;
 m=nx/ns;
 if(m<rho)
 yin(J)=-1
 j=1;
 if (yin(j)>=yin(j+1)& yin(j)>= yin(j+2)) then
 J=1;
 elseif (yin(j+1)>= yin(j) & yin(j+1)>=yin(j+2))
 J=2;
 else
 J=3;
 end
 end
for i=1:4
 temp=0;
 temp=L-1+nx;
 b(i,J)=(L*x1(i))/temp;
end
disp("b=");
disp(b)
for i=1:4
 t(J,i)=x1(i);
end
disp("t=");
disp(t)
e=e+1;
end
