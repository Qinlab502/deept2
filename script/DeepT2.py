#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import pandas as pd
import esm
import numpy
import torch
import torch.nn as nn

from Bio import SeqIO
import joblib

import argparse
import os
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from umap import UMAP
import plotly.express as px
import plotly.graph_objs as go


# In[7]:


# Run it in cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[8]:


# To match the label with the figure
labels={0:'II',1:'III',2:'VII',3:'IX',4:'VIII',5:'IV',6:'I',7:'V',8:'VI'}


# In[9]:


def get_embeddings(FASTA_PATH,EMB_PATH,EMB_LAYER):
    global headers
    headers=[]
    ys = []
    Xs = []
    for header, _seq in esm.data.read_fasta(FASTA_PATH):
        headers.append(header)
        label = header.split('|')[-1]
        if label != 'unknown':
            ys.append(int(label))
        else:
            ys.append(label)
        fn = f'{EMB_PATH}/{header}.pt'
        embs = torch.load(fn)
        Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs = torch.stack(Xs, dim=0).numpy()
    return Xs,ys


# In[10]:


esm2_2560=get_embeddings('../datasets/ksb_166_5b.fasta','../datasets/embeddings/',36)
esm2_2560_X=esm2_2560[0]
esm2_2560_X = torch.Tensor(esm2_2560_X).to(device)
annotation_file = pd.read_csv("../datasets/annotation_file_166_5b.csv")
esm2_2560_y = annotation_file['nine_class']
esm2_2560_y = esm2_2560_y.to_numpy(dtype=int)
esm2_2560_y = torch.tensor(esm2_2560_y, dtype=torch.long).to(device)


# In[11]:


def get_embeddings_set(set_type, EMB_PATH, EMB_LAYER):
    ys = []
    Xs = []
    for i, row in set_type.iterrows():
        identifier = row['identifier']
        label = row['nine_class']
        ys.append(label)
        fn = f'{EMB_PATH}/{identifier}.pt'
        embs = torch.load(fn)
        Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs = torch.stack(Xs, dim=0).numpy()
    return Xs, ys


# In[12]:


train_set = annotation_file[annotation_file.split == "train"]
test_set = annotation_file[annotation_file.split == "test"]
train_data=get_embeddings_set(train_set,'../datasets/embeddings',36)
Xs_train,ys_train=train_data[0],train_data[1]
Xs_train = torch.Tensor(Xs_train).to(device)
ys_train = torch.tensor(ys_train, dtype=torch.long).to(device)
train_dataset = torch.utils.data.TensorDataset(Xs_train, ys_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)


# In[13]:


# class a nn model using torch
class ksb(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(ksb, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(3)])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        for hidden_layer in self.hidden:
            x = hidden_layer(x)
            x = nn.ReLU()(x)
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.fc1(x)
        out_list.append(out)
        out = nn.ReLU()(out)
        for hidden_layer in self.hidden:
            out = hidden_layer(out)
            out_list.append(out)
            out = nn.ReLU()(out)
        out = self.fc2(out)
        out_list.append(out)
        return out, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.fc1(x)
        out = nn.ReLU()(out)
        if layer_index == 0:
            return out
        for i, hidden_layer in enumerate(self.hidden):
            if i == (layer_index - 1):
                out = hidden_layer(out)
                return out
            else:
                out = hidden_layer(out)
                out = nn.ReLU()(out)
        if layer_index == (len(self.hidden) + 1):
            out = self.fc2(out)
            return out


# In[14]:


# class a nn model using torch
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(3)])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.Tanh()(x)
        for hidden_layer in self.hidden:
            x = hidden_layer(x)
            x = nn.Tanh()(x)
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.fc1(x)
        out_list.append(out)
        out = nn.Tanh()(out)
        for hidden_layer in self.hidden:
            out = hidden_layer(out)
            out_list.append(out)
            out = nn.Tanh()(out)
        out = self.fc2(out)
        out_list.append(out)
        return out, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.fc1(x)
        out = nn.Tanh()(out)
        if layer_index == 0:
            return out
        for i, hidden_layer in enumerate(self.hidden):
            if i == (layer_index - 1):
                out = hidden_layer(out)
                return out
            else:
                out = hidden_layer(out)
                out = nn.Tanh()(out)
        if layer_index == (len(self.hidden) + 1):
            out = self.fc2(out)
            return out


# In[15]:


def sample_class_mean(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data=data.to(device)
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        #equal_flag = pred.eq(target.cuda()).cpu()
        equal_flag=pred.eq(target.to(device)).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label]                     = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        #temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        #temp_precision = torch.from_numpy(temp_precision).float().cuda()
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


# In[16]:


def Mahalanobis_score(model, test_loader, num_classes, outf, out_flag, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))

        
    g = open(temp_file_name, 'w')
    
    for data, target in test_loader:
        
        #data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = torch.add(data.data, -magnitude, gradient)
        
        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis


# In[17]:


def T2PK_predict(index,test_filter_X,test_filter_y,output_path,name):
    load_torch_MLP = torch.load('../model/Enhanced_T2PK_Classifier.pth',map_location=device)
    
    test_filter_y_ = np.array(test_filter_y)
    test_filter_y_[test_filter_y_ == 'unknown'] = -1
    test_filter_y_ = np.array(test_filter_y_).astype(int)
    test_filter_y_=torch.tensor(test_filter_y_, dtype=torch.long).to(device)
    
    # Create dataset from total_X_filtered and total_y_filtered
    dataset = torch.utils.data.TensorDataset(esm2_2560_X, esm2_2560_y)
    or_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create a dataset and data loader for the test set
    test_dataset = TensorDataset(test_filter_X, test_filter_y_)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # set information about feature extaction
    load_torch_MLP.eval()
    temp_x = torch.rand(len(train_loader), 2560).to(device)
    temp_x = Variable(temp_x)
    temp_x = temp_x.reshape(temp_x.shape[0], -1)
    temp_list = load_torch_MLP.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    
    # get sample mean and covariance
    sample_mean, precision = sample_class_mean(load_torch_MLP, 9, feature_list, train_loader)
    
    # Acauqiring confidence scores of unseen data 
    for i in range(num_output):
        M_unknown = Mahalanobis_score(load_torch_MLP, test_loader, 9, output_path, False, sample_mean, precision, i, 0.0)
        M_unknown = np.asarray(M_unknown, dtype=np.float32)
        if i == 0:
            Mahalanobis_unknown = M_unknown.reshape((M_unknown.shape[0], -1))
        else:
            Mahalanobis_unknown = np.concatenate((Mahalanobis_unknown, M_unknown.reshape((M_unknown.shape[0], -1))), axis=1)
        unknown_Mahalanobis = Mahalanobis_unknown
    df2 = pd.DataFrame(unknown_Mahalanobis, index=range(0, len(test_loader)), columns=["input_layer", "hidden_layer1", "hidden_layer2", "hidden_layer3", "output_layer"])   
    Out_df = df2[["hidden_layer1", "hidden_layer2", "hidden_layer3"]]
    label = pd.DataFrame({'label': [0] * len(test_filter_y)}) 
    model = joblib.load('../model/odd_detection.pkl')
    cols = ['hidden_layer1']
    label['isolation_score'] = model.decision_function(Out_df[cols])
    label['isolation_predict'] = model.predict(Out_df[cols])
    
    #Get output file
    with open(output_path+'/'+name+'.txt','w') as f1:
        f1.write('Predicted abnormal ksb header:'+'\n')
        if -1 not in label['isolation_predict'].values:
            f1.write("No abnormal ksb found in test"+'\n')
        else:
            test_out=label[label['isolation_predict'] == -1].index.tolist()
            for i in range(len(test_out)):
                f1.write(headers[index[test_out[i]]].split('|')[1]+'\n')
                
        test_in=label[label['isolation_predict'] == 1].index.tolist()
        with torch.no_grad():
            test_ys_pred = load_torch_MLP(test_filter_X[test_in])
            _, ys_pred = torch.max(test_ys_pred, 1) # get max of each row
            ys_pred = ys_pred.cpu()
         
        f1.write('Predicted in_distribution ksb class:'+'\n')
        test_filter_X_=test_filter_X[test_in]
        test_filter_y_=[test_filter_y[i] for i in test_in]
        for i in range(len(test_in)):
            test_euc=eudi_figure(index,test_filter_X_[i],test_filter_y_[i],output_path,name,ys_pred[i])
            f1.write(str(i)+'\t'+headers[index[test_in[i]]].split('|')[1]+'\t'+str(labels[ys_pred.tolist()[i]])+'\n')
            for data in range(len(test_euc)):
                f1.write('\t'+test_euc.iloc[data,0].split('|')[1]+'\t'+str(test_euc.iloc[data,1])+'\n')
    return ys_pred


# In[18]:


def umap_reduce(embeddings, **kwargs):
    """Wrapper around :meth:`umap.UMAP` with defaults for bio_embeddings"""
    umap_params = dict()

    umap_params['n_components'] = kwargs.get('n_components', 3)
    umap_params['min_dist'] = kwargs.get('min_dist', .01)
    umap_params['spread'] = kwargs.get('spread', 1)
    umap_params['random_state'] = kwargs.get('random_state', 420)
    umap_params['n_neighbors'] = kwargs.get('n_neighbors', 10)
    umap_params['verbose'] = kwargs.get('verbose', 1)
    umap_params['metric'] = kwargs.get('metric', 'cosine')

    transformed_embeddings = UMAP(**umap_params).fit_transform(embeddings)

    return transformed_embeddings


# In[19]:


def eudi_figure(index,test_filter_X,test_filter_y,output_path,name,ys_pred):
    # extract the predicted class embeddings
    annotation_file=pd.read_csv('../datasets/annotation_file_166_5b.csv')
    pre_class=ys_pred.numpy().tolist()
    index=annotation_file[annotation_file.loc[:,'nine_class']==pre_class].index.tolist()
    test_in_X=esm2_2560_X[index]
    test_in_y=esm2_2560_y[index]
    # Add G7 in extracted embeddings
    test_all_X=torch.cat((test_in_X,test_filter_X.reshape(1,2560)),0)
    test_all_y=np.append(test_in_y.cpu(),np.array(test_filter_y))
    
    # Get euclidean distance
    test_all_umap=umap_reduce(test_all_X.cpu(),options={'perplexity': 6, 'n_iter': 15000})
    test_euc_list=[]
    for x in test_all_umap[:-1]:
        test_euc_list.append(np.sqrt(np.sum((test_all_umap[-1]-x)**2)))
    test_euc=pd.DataFrame(zip(annotation_file.iloc[index,0],test_euc_list),columns=['identifier','distance'])
    like=test_euc[test_euc.loc[:,'distance']==test_euc.loc[:,'distance'].min()]
    
    test_euc.loc[len(test_euc.index)]=['test unknown',0]
    test_euc['component_0']=test_all_umap[:, 0]
    test_euc['component_1']=test_all_umap[:, 1]
    test_euc['component_2']=test_all_umap[:, 2]
    test_euc.sort_values(by="distance" , inplace=True, ascending=True) 
    test_euc.to_csv(output_path+'/'+name+'_'+labels[pre_class]+'_unknown.csv',index=None)
    
    # Get distance figure
    test_known=test_euc.iloc[1:,:]
    test_unknown=test_euc.iloc[0,:]
    
    # create trace for known points
    known_trace = go.Scatter3d(
        x=test_known['component_0'],
        y=test_known['component_1'],
        z=test_known['component_2'],
        mode='markers',
        marker=dict(
            size=5,
            color='#1874CD',
            opacity=0.8
        ),
        hovertext=test_known.loc[:,'identifier']
    )

    # create trace for unknown point
    unknown_trace = go.Scatter3d(
        x=[test_unknown['component_0']],
        y=[test_unknown['component_1']],
        z=[test_unknown['component_2']],
        mode='markers',
        marker=dict(
            size=6,
            color='#DC143C',
            opacity=0.8
        ),
        hovertext=test_unknown['identifier']
    )
    
    # create traces for lines connecting unknown point to known points
    lines = {'x': [], 'y': [], 'z': []}
    
    for i in range(len(test_known)):
        lines['x'] += [list([test_unknown['component_0'], test_known['component_0'][i], None])]
        lines['y'] += [list([test_unknown['component_1'], test_known['component_1'][i], None])]
        lines['z'] += [list([test_unknown['component_2'], test_known['component_2'][i], None])]

    line_traces = []
    for i in range(len(lines['x'])):
        line_trace = go.Scatter3d(
            x=lines['x'][i],
            y=lines['y'][i],
            z=lines['z'][i],
            mode='lines',
            line=dict(
                color='#C5987C',
                width=4,
                dash='dash'
            ),
            hoverinfo='none',
            showlegend=False
        )
        line_traces.append(line_trace)

    # plot the figure
    fig = go.Figure(data=[known_trace, unknown_trace] + line_traces)
    
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        legend=dict(
            x=0,
            y=1,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
        ),
        template='plotly_white'
    )
    
    fig.write_html(output_path+'/'+name+'_'+labels[pre_class]+'.html')
    
    return test_euc.iloc[1:4,:]


# In[20]:


def ksb_predict(fasta_file,embeddings,output_path,name):
    ksb=torch.load('../model/KSb_binary_Classifier.pt',map_location=device)
    
    test=get_embeddings(fasta_file,embeddings,36)
    test_X,test_y=test[0],test[1]
    test_X = torch.Tensor(test_X).to(device)
    
    # Make predictions on the test set
    with torch.no_grad():
        test_ys_pred = ksb(test_X)
        _, test_ys_pred = torch.max(test_ys_pred, 1) # get max of each row
        test_ys_pred = test_ys_pred.cpu()
        
    # Filter the embeddings that had been predicted into the positive label
    index=[index for index,bools in enumerate(test_ys_pred==0) if bools]
    if index:
        test_filter_X=test_X[index]
        test_filter_y=[test_y[i] for i in index]
    else:
        print('There is no ksb in the genome')
    mkdir(output_path)
    T2PK_predict(index,test_filter_X,test_filter_y,output_path,name)
    
    return 0


# In[21]:


def mkdir(output_path):
    folder = os.path.exists(output_path)
    if not folder:
        os.makedirs(output_path)


# In[22]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type = str, help = "Input fasta file")
    parser.add_argument("--embedding",type = str, help = "Input embeddings")
    parser.add_argument("--output",type = str, help = "Custom Output directory path")
    parser.add_argument("--name",type = str, help = "Custom Output filename")
    
    args = parser.parse_args()
    
    output_path=args.output
    name=args.name
    
    ksb_predict(args.fasta,args.embedding,output_path,name)


# In[ ]:




