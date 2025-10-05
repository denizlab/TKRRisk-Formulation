import time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score,log_loss


def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    y_pred = y_preds.numpy()
    y_true = y_targets.numpy()

    #print(y_pred)
    #print(y_true)

    #print(y_pred.shape,y_true.sum())
    
    return roc_auc_score(y_true, y_pred)


def train(train_loader, val_loader,model,optimizer,scheduler,loss_fn,experiment,result_path,dropout,num_epoch=2, print_every=100,device="cpu"):
    # Training steps
    start_time = time.time()
    
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    best_acc = 0
    best_auc = 0
    predictions = []
    truths = []
    wait=0

    best_val_loss = np.inf
    step=-1
    
    for epoch in range(num_epoch):
        

        correct = 0
        total = 0
        predictions = []
        truths = []
        print('Epoch: {}/{}'.format(epoch, num_epoch-1))
        #h = model.init_hidden(batch_size)
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            #h=h.data
            data, labels = data.to(device), labels.to(device)
            
            #print(data)
            
            optimizer.zero_grad()
            outputs = model(data)
            labels = labels.type_as(outputs)
            pred = torch.tensor(outputs>=0).squeeze()
            #print(pred,outputs,labels)
            
            
            #print('pred', pred, type(pred))
            #print('labels',labels)

            
            model.zero_grad()
            step+=1
            #print(outputs,labels)
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            experiment.log_metric('loss_train', loss.item(), step=step)
            experiment.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=step)
            total += labels.size(0)
            correct += float((pred.cpu().numpy() == labels.cpu().numpy()).sum())
            truths.extend(Variable(labels))
            predictions.extend(outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()

             # report performance
            if (i + 1) % print_every == 0:
                print('Train set | epoch: {:3d} | {:6d}/{:6d} batches | Loss: {:6.4f}'.format(
                    epoch, i + 1, len(train_loader), loss.item()))
                
                
        acc = (100 * correct / total)
        acc_train.append(float(acc))    
        loss_train.append(loss.item())
        #auc_train = roc_auc_compute_fn(torch.sigmoid(predictions), truths)
        auc_train = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
        experiment.log_metric('auc_train', auc_train, step=step)

        experiment.log_metric('acc_train', acc, step=step)
        #auc_train = roc_auc_compute_fn(torch.sigmoid(torch.tensor(pred)),torch.tensor(truths))
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        print('Train set | Accuracy: {:6.4f} |  time elapse: {:>9}'.format(acc, elapse))
        print('AUC train : ',auc_train)
        print('Loss train epoch: ',loss.item())
             
    
    # Evaluate after every epoch
        correct = 0
        total = 0
        model.eval()

        predictions = []
        truths = []
        losses_val = []

        with torch.no_grad():
            for i, (data, labels) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                labels = labels.type_as(outputs)
                pred = (outputs>=0).squeeze()
                
                
                
                #print(outputs)
                
                loss = loss_fn(outputs.squeeze(), labels.squeeze())
                losses_val.append(loss.item())
                
                total += labels.size(0)
                correct += float((pred.cpu().numpy() == labels.cpu().numpy()).sum())
                #predictions.extend(outputs)
                #truths.extend(Variable(labels))
                predictions.extend(outputs)
                truths.extend(Variable(labels))
                correct = int(correct)
                
            acc = (100 * correct / total)
            auc_val = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
            acc_val.append(float(acc))
            experiment.log_metric('auc_val', auc_val, step=step)

            experiment.log_metric('acc_val', acc, step=step)
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Validation set | epoch: {:3d} | {:6d}/{:6d} batches | Loss: {:6.4f}'.format(epoch, i + 1, len(val_loader), loss.item()))
            print('Validation set | Accuracy: {:6.4f} |  time elapse: {:>9}'.format(acc, elapse))
            print('AUC val : ',auc_val)
            loss_val = np.mean(losses_val)

            # save models
            
            wait+=1
            if auc_val > best_auc:
                best_auc = max(auc_val, best_auc)
                wait=0
                
                full_path = result_path + "best_auc_model.prm"
                torch.save(model.state_dict(), full_path)
            if loss_val <best_val_loss:
                best_val_loss = min(loss_val, best_val_loss)
                wait=0
                
                full_path = result_path + "best_loss_model.prm"
                torch.save(model.state_dict(), full_path)
            print('BEST AUC :',best_auc,', wait: ',wait)
            

    #model.load_state_dict(best_model_wts)

    return model, loss_train, loss_val, acc_train, acc_val,best_auc



    #model.load_state_dict(best_model_wts)

    return model, loss_train, loss_val, acc_train, acc_val,best_auc

def train_siam(train_loader, val_loader,model,optimizer,scheduler,loss_fn,loss_fn_1,experiment,gamma,result_path,dropout,num_epoch=2, print_every=100,device="cpu"):
    # Training steps
    start_time = time.time()
    
    loss_train = []
    loss_val = []
    best_auc = 0
    best_auc_1 = 0
    predictions = []
    truths = []
    wait=0

    best_val_loss = np.inf
    best_siam_loss = np.inf
    step=-1
    
    for epoch in range(num_epoch):
        

        correct = 0
        total = 0
        train_siam_loss = 0
        train_loss = 0
        predictions = []
        predictions_1 = []
        predictions_0_1 = []
        truths_1 = []
        truths = []
        print('Epoch: {}/{}'.format(epoch, num_epoch-1))
        #h = model.init_hidden(batch_size)
        model.train()
        for i, (data, labels1,labels2,siam_labels) in enumerate(train_loader):
            #h=h.data
            data, labels1,labels2,siam_labels = data.to(device), labels1.to(device),labels2.to(device),siam_labels.to(device)
            
            #print(data)
            
            optimizer.zero_grad()
            out1= model(data[:,0,:,:,:])
            out2= model(data[:,1,:,:,:])
            labels1 = labels1.type_as(out1)
            labels2 = labels2.type_as(out1)
            siam_labels = siam_labels.type_as(out1)
            #print(pred,outputs,labels)
            
            
            #print('pred', pred, type(pred))
            #print('labels',labels)

            
            model.zero_grad()
            step+=1
            #print(outputs,labels)
            # s_loss = torch.log
            s_loss = loss_fn(torch.log(torch.sigmoid(out1[(siam_labels!=-1).squeeze()]).squeeze()),
                torch.log(torch.sigmoid(out2[(siam_labels!=-1).squeeze()]).squeeze()),
                 -1*torch.ones_like(out1[(siam_labels!=-1).squeeze()]).squeeze())
                
            if ((labels2==-1)*1.0).mean()!=1:
                loss = (loss_fn_1(out1.squeeze(), labels1.squeeze())+loss_fn_1(out2[(labels2!=-1).squeeze()].squeeze(), labels2[(labels2!=-1).squeeze()].squeeze())) + gamma*s_loss
                try:
                    truths.extend(labels2[(labels2!=-1).squeeze()])
                    predictions.extend(out2[(labels2!=-1).squeeze()].squeeze())
                except:
                    truths.append(labels2[(labels2!=-1).squeeze()])
                    predictions.append(out2[(labels2!=-1).squeeze()].squeeze())
            else:
                loss = loss_fn_1(out1.squeeze(), labels1.squeeze()) + gamma*s_loss

            train_siam_loss+=s_loss.item()*labels1.size(0)
            train_loss+=loss.item()*labels1.size(0)
            
            #experiment.log_metric('step_loss_risk_train', s_loss.item(), step=step)
            #experiment.log_metric('loss_train', loss.item(), step=step)
            experiment.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=step)
            total += labels1.size(0)
            predictions_0_1.extend(out1.squeeze())
            truths_1.extend(Variable(labels1))

            loss.backward()
            optimizer.step()
            scheduler.step()

             # report performance
            if (i + 1) % print_every == 0:
                print('Train set | epoch: {:3d} | {:6d}/{:6d} batches | Loss: {:6.4f}'.format(
                    epoch, i + 1, len(train_loader), loss.item()))
                
                
        loss_train.append(loss.item())
        predictions.extend(predictions_0_1)
        truths.extend(truths_1)
        auc_train_1 = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
        #auc_train = roc_auc_compute_fn(torch.sigmoid(predictions), truths)
        experiment.log_metric('epoch_risk_loss_train', train_siam_loss/total, step=step)
        experiment.log_metric('epoch_loss_train', train_loss/total, step=step)
        experiment.log_metric('auc_train', auc_train_1, step=step)

        #auc_train = roc_auc_compute_fn(torch.sigmoid(torch.tensor(pred)),torch.tensor(truths))
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        print('Loss train epoch: ',loss.item())
             
    
    # Evaluate after every epoch
        correct = 0
        total = 0
        model.eval()

        predictions = []
        truths = []
        losses_val = []
        predictions_1 = []
        predictions_0_1 = []
        truths_1 = []
        val_loss = 0
        val_siam_loss = 0

        with torch.no_grad():
            for i, (data, labels1,labels2,siam_labels) in enumerate(val_loader):
                data, labels1,labels2,siam_labels = data.to(device), labels1.to(device),labels2.to(device),siam_labels.to(device)
            
                #print(data)
                
                out1= model(data[:,0,:,:,:])
                out2= model(data[:,1,:,:,:])
                labels1 = labels1.type_as(out1)
                labels2 = labels2.type_as(out1)
                siam_labels = siam_labels.type_as(out1)
                #print(pred,outputs,labels)
                
                
                #print('pred', pred, type(pred))
                #print('labels',labels)

                
                step+=1
                #print(outputs,labels)
                s_loss = loss_fn(torch.sigmoid(out1[(siam_labels!=-1).squeeze()]).squeeze(),torch.sigmoid(out2[(siam_labels!=-1).squeeze()]).squeeze(), -1*torch.ones_like(out1[(siam_labels!=-1).squeeze()]).squeeze())
                
                if ((labels2==-1)*1.0).mean()!=1:
                    loss = (loss_fn_1(out1.squeeze(), labels1.squeeze())+loss_fn_1(out2[(labels2!=-1).squeeze()].squeeze(), labels2[(labels2!=-1).squeeze()].squeeze())) + gamma*s_loss
                    try:
                        truths.extend(labels2[(labels2!=-1).squeeze()])
                        predictions.extend(out2[(labels2!=-1).squeeze()].squeeze())
                    except:
                        truths.append(labels2[(labels2!=-1).squeeze()])
                        predictions.append(out2[(labels2!=-1).squeeze()].squeeze())
                else:
                    loss = loss_fn_1(out1.squeeze(), labels1.squeeze()) + gamma*s_loss

                val_siam_loss+=s_loss.item()*labels1.size(0)
                val_loss+=loss.item()*labels1.size(0)
                total += labels1.size(0)
                predictions_0_1.extend(out1.squeeze())
                truths_1.extend(Variable(labels1))
                
            predictions.extend(predictions_0_1)
            truths.extend(truths_1)
            auc_val_1 = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
            experiment.log_metric('auc_val', auc_val_1, step=step)

            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Validation set | epoch: {:3d} | {:6d}/{:6d} batches | Loss: {:6.4f}'.format(epoch, i + 1, len(val_loader), loss.item()))
            print('AUC val : ',auc_val_1)

            # save models
            
            wait+=1
            if auc_val_1 > best_auc:
                best_auc = max(auc_val_1, best_auc)
                wait=0
                torch.save(model.state_dict(), result_path + "best_auc_model.prm")
            if best_siam_loss > val_siam_loss/total:
                best_siam_loss = min(val_siam_loss/total, best_siam_loss)
                wait=0
                
                torch.save(model.state_dict(), result_path + "best_risk_loss_model.prm")
            if val_loss/total <best_val_loss:
                best_val_loss = min(val_loss/total, best_val_loss)
                wait=0
                
                torch.save(model.state_dict(), result_path + "best_loss_model.prm")
            print('BEST AUC :',best_auc,', wait: ',wait)
            print("Best Val loss ",best_val_loss)
            print("Best risk reg loss ",best_siam_loss)
            

    #model.load_state_dict(best_model_wts)

    return model, loss_train, val_loss,best_auc


def evaluate(model,val_loader,loss_fn,device):
    correct = 0
    total = 0
    model.eval()

    predictions = []
    truths = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)

            if labels.size(0)==0:
                continue

            outputs = model(data)
            labels = labels.type_as(outputs)
            pred = (outputs.data>=0).squeeze()

            #print(outputs)



            loss = loss_fn(outputs.squeeze(), labels)

            total += labels.size(0)
            correct += (pred == labels).sum()
            #predictions.extend(outputs)
            #truths.extend(Variable(labels))
            predictions.extend(outputs)
            truths.extend(Variable(labels))
            correct = int(correct)

        acc = (100 * correct / total)
        print(len(predictions),len(truths))
        auc_val = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
        
        return auc_val,acc
def train_all_base(train_loader, val_loader,yr,val_df,model,optimizer,scheduler,loss_fn,experiment,result_path,dropout,num_epoch=2, print_every=100,device="cpu"):
    # Training steps
    start_time = time.time()
    
    loss_train = []
    loss_val = []
    best_auc = 0
    predictions = []
    truths = []
    wait=0

    best_val_loss = np.inf
    step=-1

    print(model)
    
    for epoch in range(num_epoch):
        

        correct = 0
        total = 0
        predictions = []
        truths = []
        print('Epoch: {}/{}'.format(epoch, num_epoch-1))
        #h = model.init_hidden(batch_size)
        model.train()
        for i, (data, labels1,labels2,siam_labels) in enumerate(train_loader):
            #h=h.data
            data, labels1,labels2,siam_labels = data.to(device), labels1.to(device),labels2.to(device),siam_labels.to(device)
            
            #print(data)
            
            optimizer.zero_grad()
            all_data = torch.cat([data[:,0,:,:,:],data[(labels2!=-1).squeeze(),1,:,:,:]],0)

            #print(all_data.shape,"all data shape")
            all_labels = torch.cat([labels1,labels2[(labels2!=-1).squeeze()]],0)
            #print("here")
            output= model(all_data)
            #print("here 2")
            all_labels = all_labels.type_as(output)
            
            #print(pred,outputs,labels)
            
            
            #print('pred', pred, type(pred))
            #print('labels',labels)

            
            model.zero_grad()
            step+=1
            #print(outputs,labels)
            loss = loss_fn(output.squeeze(), all_labels.squeeze())
            experiment.log_metric('loss_train', loss.item(), step=step)
            experiment.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=step)
            truths.extend(Variable(all_labels))
            predictions.extend(output.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()

             # report performance
            if (i + 1) % print_every == 0:
                print('Train set | epoch: {:3d} | {:6d}/{:6d} batches | Loss: {:6.4f}'.format(
                    epoch, i + 1, len(train_loader), loss.item()))
                
                
        loss_train.append(loss.item())
        #auc_train = roc_auc_compute_fn(torch.sigmoid(predictions), truths)
        auc_train = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions)),torch.tensor(truths))
        experiment.log_metric('auc_train', auc_train, step=step)

        #auc_train = roc_auc_compute_fn(torch.sigmoid(torch.tensor(pred)),torch.tensor(truths))
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        print('AUC train : ',auc_train)
        print('Loss train epoch: ',loss.item())
             
    
    # Evaluate after every epoch
        _,predictions_0_1,truths_0,predictions_1,truths = evaluate_all_base(model,val_loader,device)

        val_df[yr+"Pred_0"] = torch.tensor(predictions_0_1).cpu().numpy()
        val_df[yr+"Pred_1"] = torch.tensor(predictions_1).cpu().numpy()

        val_loss,auc_val = get_metrics(val_df,yr)
        experiment.log_metric('auc_val', auc_val, step=step)
            
            
        wait+=1
        if auc_val > best_auc:
            best_auc = max(auc_val, best_auc)
            wait=0
            
            full_path = result_path + "best_auc_model.prm"
            torch.save(model.state_dict(), full_path)
        if val_loss <best_val_loss:
            best_val_loss = min(val_loss, best_val_loss)
            wait=0
            
            full_path = result_path + "best_loss_model.prm"
            torch.save(model.state_dict(), full_path)
        print('BEST AUC :',best_auc,', wait: ',wait)

        if wait>40:
            return
            

    #model.load_state_dict(best_model_wts)

    return model, loss_train, loss_val,best_auc


def evaluate_siam(model,val_loader,device):
    correct = 0
    total = 0
    model.eval()

    predictions = []
    truths = []
    losses_val = []
    predictions_1 = []
    truths_0 = []
    predictions_0_1 = []

    with torch.no_grad():
        for i, (data, labels1,labels2,siam_labels) in enumerate(val_loader):
            data, labels1,labels2,siam_labels = data.to(device), labels1.to(device),labels2.to(device),siam_labels.to(device)
        
            #print(data)
            
            out1,out2,hid1,hid2 = model(data)
            labels1 = labels1.type_as(out1)
            labels2 = labels2.type_as(out1)
            siam_labels = siam_labels.type_as(out1)
            #print(pred,outputs,labels)
            
            
            #print('pred', pred, type(pred))
            #print('labels',labels)

            
            #print(outputs,labels)
            
            total += labels1.size(0)
            truths.extend(Variable(labels2))
            truths_0.extend(Variable(labels1))
            predictions.extend((hid2.squeeze() - hid1.squeeze()).pow(2).sum(1))
            predictions_1.extend(out2)
            predictions_0_1.extend(out1)

        #auc_val = roc_auc_compute_fn(torch.tensor(predictions),torch.tensor(truths))
        #auc_val_1 = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions_0_1)),torch.tensor(truths))


        #predictions_1.extend(predictions_0_1)
        #truths.extend(truths_0)

        #all_auc = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions_1)),torch.tensor(truths))
        
    return 0,0,predictions_0_1,truths_0,predictions_1,truths
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_metrics(val_df,yr):
    auc_pred = []
    auc_truth = []
    for i in range(val_df.shape[0]):
        loc = val_df.iloc[i]
        if loc["last_month"]=="00m":
            auc_pred.append(sigmoid(loc[yr+"Pred_0"]))
            auc_truth.append((loc[yr+"Label0"]))
        else:
            auc_pred.extend([sigmoid(loc[yr+"Pred_0"]),sigmoid(loc[yr+"Pred_1"])])
            auc_truth.extend([loc[yr+"Label0"],loc[yr+"Label1"]])



    auc_pred = np.array(auc_pred)
    auc_truth = np.array(auc_truth)
    val_auc = roc_auc_score(auc_truth,auc_pred)
    val_loss = log_loss(auc_truth,sigmoid(auc_pred))

    return val_loss,val_auc


def evaluate_all_base(model,val_loader,device):
    correct = 0
    total = 0
    model.eval()

    predictions = []
    truths = []
    losses_val = []
    predictions_1 = []
    truths_0 = []
    predictions_0_1 = []

    with torch.no_grad():
        for i, (data, labels1,labels2,siam_labels) in enumerate(val_loader):
            data, labels1,labels2,siam_labels = data.to(device), labels1.to(device),labels2.to(device),siam_labels.to(device)

            #print(data)

            out1 = model(data[:,0,:,:,:])
            out2 = model(data[:,1,:,:,:])
            labels1 = labels1.type_as(out1)
            labels2 = labels2.type_as(out1)
            siam_labels = siam_labels.type_as(out1)
            #print(pred,outputs,labels)


            #print('pred', pred, type(pred))
            #print('labels',labels)


            #print(outputs,labels)

            total += labels1.size(0)
            truths.extend(Variable(labels2))
            truths_0.extend(Variable(labels1))
            predictions_1.extend(out2)
            predictions_0_1.extend(out1)


        
        auc_val_1 = roc_auc_compute_fn(torch.sigmoid(torch.tensor(predictions_0_1)),torch.tensor(truths_0))


        
    return auc_val_1,predictions_0_1,truths_0,predictions_1,truths
    

