# importing required modules  
import matplotlib.pyplot as plt 
  
# creating plotting data 
xaxis =[1, 4, 9, 16, 25, 36, 49, 64, 81, 100] 
yaxis =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
  
# plotting  
plt.plot(xaxis, yaxis) 
plt.xlabel("X") 
plt.ylabel("Y") 
  
# saving the file.Make sure you  
# use savefig() before show(). 
plt.savefig("squares.png") 
 
plt.show()