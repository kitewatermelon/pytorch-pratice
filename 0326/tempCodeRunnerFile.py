
# lr = 1e-3
# optim = Adam(model.parameters(),lr=lr)
# for epoch in range(20):
#     for data, label in train_loader:
#         optim.zero_grad()
#         data = torch.reshape(data, (-1, 784)).to(device)
#         preds = model(data)
        
#         loss = nn.CrossEntropyLoss()(preds, label.to(device))
#         loss.backward()
#         optim.step()
        
#     print(f'epoch{epoch+1} loss:{loss.item()}')
    
# torch.save(model.state_dict(), "MNIST.pth")
