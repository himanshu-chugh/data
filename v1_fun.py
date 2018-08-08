import torch 
import pandas as pd
from math import e
def V1():
    

    dtype = torch.float64
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    #N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold input and outputs.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Tensors during the backward pass.
    #x = torch.randn(N, D_in, device=device, dtype=dtype)
    #y = torch.randn(N, D_out, device=device, dtype=dtype)



    print("Enter DATASET csv_file name :\n")
    file_name = input()
    df=pd.read_csv(file_name)
    print("is there any missing values inn dataset ? y/n \n\t")
    ans=input()
    if(ans=='y'):
        print("do you want to fill them with a values? y/n\nelse it would be filled with 0")
        ans2=input()
        if(ans2=='y'):
            value_for_nan=int(input("Enter value"))
            df=df.fillna(value_for_nan)
        else:
            df=df.fillna(0)

    
    print("\nHere are all the attribuites :\n\t",df.keys(),"\n\t")
    while(True):
        target=input("Enter target key plz type exact : ")
        if(target in df.keys().values ):

            break
        else:
            print("\n\t you Entered wrong key \n\t plz try again...!!")
    y=torch.tensor(df[target].values,device=device,dtype=dtype)
    df=df.drop(columns=[target])
    x=torch.tensor(df.values,device=device,dtype=dtype)
    #x = torch.randn(N, D_in, device=device, dtype=dtype)
    #torch_tensor = torch.tensor(targets_df['targets'].values)

    #N, D_in, H, D_out = 64, 1000, 100, 10

    print("\n\tIN this NN there are 3 hidden layers u can size them if u want y/n else it will be a default 6x6x6")
    ans3=input()
    if(ans3=='y'):
        L2_size=int(input("input layer 2 size including baies term : "))
        L3_size=int(input("input layer 3 size including baies term : "))
        L4_size=int(input("input layer 4 size including baies term : "))
    else:
        L2_size=6
        L3_size=6
        L4_size=6
        

    D_in=len(x[0])

    w1 = torch.randn(D_in, L2_size, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(L2_size, L3_size, device=device, dtype=dtype, requires_grad=True)
    w3 = torch.randn(L3_size, L4_size, device=device, dtype=dtype, requires_grad=True)
    w4 = torch.randn(L4_size, 1, device=device, dtype=dtype, requires_grad=True)






    # Create random Tensors for weights.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    #w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    #w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    print("\n\tWant to enter learning rate and no of epoch ?")
    ans4=input("y/n")
    if(ans4=='y'):
        learning_rate=float(input("learning rate : "))
        epoch=int(input("epochs : "))
    else:
        learning_rate=.1
        epoch=10000

    for t in range(epoch):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        L2 = x.mm(w1).clamp(min=0)
        L2p=(1/(1+(e**L2)))
        L3 = L2p.mm(w2).clamp(min=0)
        L3p=(1/(1+(e**L3)))
        L4 = L3.mm(w3).clamp(min=0)
        L4p=(1/(1+(e**L4)))
        y_pred = L4p.mm(w4).clamp(min=0)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        """
        loss.backward()
        print(w1)
        print(w2)
        print(w3)
        print(w4)
        print(loss)
        print(w1.grad)
        print(w2.grad)
        print(w3.grad)
        print(w4.grad)

        """
        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w3 -= learning_rate * w3.grad
            w4 -= learning_rate * w4.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()
            w4.grad.zero_()
