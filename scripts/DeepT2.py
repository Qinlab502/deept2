import numpy as np
import pandas as pd
import esm
import torch
import torch.nn as nn
import joblib
import argparse
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from umap import UMAP
import plotly.graph_objs as go
from MDS import get_MDS
import h5py

# Run it in cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get embedding from import fasta file and embedding output
def get_embeddings(FASTA_PATH, EMB_PATH, EMB_LAYER):
    global headers
    headers = []
    ys = []
    Xs = []
    for header, _seq in esm.data.read_fasta(FASTA_PATH):
        headers.append(header)
        fn = f'{EMB_PATH}/{header}.pt'
        embs = torch.load(fn)
        Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs = torch.stack(Xs, dim=0).numpy()
    return Xs

def read_h5(h5_file,annotation_file):
    embeddings=list()
    label=list()
    annotation=pd.read_csv(annotation_file)
    with h5py.File(h5_file,'r') as f:
        for identifier in annotation.iloc[:,0]:
            embeddings.append(np.array(f[identifier]))
            label.append(f[identifier].attrs["label"])
    embeddings=np.array(embeddings)
    
    return embeddings,label

esm2_2560_X,esm2_2560_y=read_h5('./datasets/esm2_2560_embeddings.h5','./datasets/annotation_file.csv')
esm2_2560_X = torch.Tensor(esm2_2560_X).to(device)
esm2_2560_y = torch.tensor(esm2_2560_y, dtype=torch.long).to(device)

# To match the label with the figure
labels = {0: 'II', 1: 'III', 2: 'VII', 3: 'IX', 4: 'VIII', 5: 'IV', 6: 'I', 7: 'V', 8: 'VI'}

# class ksb model using torch
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

# class T2PK model using torch
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

def ksb_predict(fasta_file, embeddings, output_path, prefix):
    ksb = torch.load('./model/KSb_binary_Classifier.pt', map_location=device)

    test_X = get_embeddings(fasta_file, embeddings, 36)
    test_X = torch.Tensor(test_X).to(device)
    test_y=np.empty(len(headers))
    test_y[:]=-1

    # Make predictions on the test set
    with torch.no_grad():
        test_ys_pred = ksb(test_X)
        _, test_ys_pred = torch.max(test_ys_pred, 1)  # get max of each row
        test_ys_pred = test_ys_pred.cpu()

    # Filter the embeddings that had been predicted into the positive label
    index = [index for index, bools in enumerate(test_ys_pred == 0) if bools]
    if index:
        test_filter_X = test_X[index]
        test_filter_y = test_y[index]
    else:
        print('There is no ksb in the genome')
    mkdir(output_path)
    T2PK_predict(index, test_filter_X, test_filter_y, output_path, prefix)

    return 0

def T2PK_predict(index, test_filter_X, test_filter_y, output_path, prefix):
    load_torch_MLP = torch.load('./model/Enhanced_T2PK_Classifier.pt', map_location=device)

#     test_filter_y_ = np.array(test_filter_y)
#     test_filter_y_[test_filter_y_ == 'unknown'] = -1
    test_filter_y_ = np.array(test_filter_y).astype(int)
    test_filter_y_ = torch.tensor(test_filter_y, dtype=torch.long).to(device)

    # Create a dataset and data loader for the test set
    test_dataset = TensorDataset(test_filter_X, test_filter_y_)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # set information about feature extaction
    load_torch_MLP.eval()
    temp_x = torch.rand(132, 2560).to(device)
    temp_x = Variable(temp_x)
    temp_x = temp_x.reshape(temp_x.shape[0], -1)
    temp_list = load_torch_MLP.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    
    #preset class get_MDS
    model = load_torch_MLP
    num_classes = 9 
    mds = get_MDS(model, num_classes, feature_list)

    # get sample mean and covariance
    if device.type == "cuda":
        sample_mean = torch.load('./model/sample_mean_gpu.pt')
        precision = torch.load('./model/precision_gpu.pt')
    else:
        sample_mean = torch.load('./model/sample_mean_cpu.pt')
        precision = torch.load('./model/precision_cpu.pt')

    # Acauqiring confidence scores of unseen data
    for i in range(num_output):
        M_unknown = mds.Mahalanobis_score(test_loader, output_path, False, sample_mean, precision, i, 0.0)
        M_unknown = np.asarray(M_unknown, dtype=np.float32)
        if i == 0:
            Mahalanobis_unknown = M_unknown.reshape((M_unknown.shape[0], -1))
        else:
            Mahalanobis_unknown = np.concatenate((Mahalanobis_unknown, M_unknown.reshape((M_unknown.shape[0], -1))), axis=1)

    df2 = pd.DataFrame(Mahalanobis_unknown, index=range(0, len(test_loader)),
                       columns=["input_layer", "hidden_layer1", "hidden_layer2", "hidden_layer3", "output_layer"])
    Out_df = df2[["hidden_layer1", "hidden_layer2", "hidden_layer3"]]
    label = pd.DataFrame({'label': [0] * len(test_filter_y)})
    model = joblib.load('./model/odd_detection.pkl')
    cols = ['hidden_layer1']
    label['isolation_score'] = model.decision_function(Out_df[cols])
    label['isolation_predict'] = model.predict(Out_df[cols])

    # Get output file
    with open(output_path + '/' + prefix + '.txt', 'w') as f1:
        f1.write('Predicted novelty ksb-T2PK:' + '\n')
        if -1 not in label['isolation_predict'].values:
            f1.write("No novelty ksb-T2PK found from input" + '\n')
        else:
            test_out = label[label['isolation_predict'] == -1].index.tolist()
            for i in range(len(test_out)):
                f1.write(headers[index[test_out[i]]][1] + '\n')

        test_in = label[label['isolation_predict'] == 1].index.tolist()
        with torch.no_grad():
            test_ys_pred = load_torch_MLP(test_filter_X[test_in])
            _, ys_pred = torch.max(test_ys_pred, 1)  # get max of each row
            ys_pred = ys_pred.cpu()
            
        f1.write('Found ' + str(len(test_in)) + ' in_distribution ksb sequence(s)' + '\n')
        f1.write('Predicted in_distribution T2PK class:' + '\n')
        test_filter_X_ = test_filter_X[test_in]
        test_filter_y_ = test_filter_y[test_in]
        for i in range(len(test_in)):
            test_euc = eudi_figure(index, test_filter_X_[i], test_filter_y_[i], output_path, prefix, ys_pred[i])
            f1.write(str(i) + '\t' + headers[index[test_in[i]]] + '\t' + str(
                labels[ys_pred.tolist()[i]]) + '\n')
            for data in range(len(test_euc)):
                f1.write('\t' + test_euc.iloc[data, 0].split('|')[1] + '\t' + str(test_euc.iloc[data, 1]) + '\n')
    return ys_pred


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


def eudi_figure(index, test_filter_X, test_filter_y, output_path, prefix, ys_pred):
    # extract the predicted class embeddings
    annotation_file = pd.read_csv('./datasets/annotation_file.csv')
    pre_class = ys_pred.numpy().tolist()
    index = annotation_file[annotation_file.loc[:, 'nine_class'] == pre_class].index.tolist()
    test_in_X = esm2_2560_X[index]
    test_in_y = esm2_2560_y[index]
    # Add G7 in extracted embeddings
    test_all_X = torch.cat((test_in_X, test_filter_X.reshape(1, 2560)), 0)
    test_all_y = np.append(test_in_y.cpu(), np.array(test_filter_y))

    # Get euclidean distance
    test_all_umap = umap_reduce(test_all_X.cpu(), options={'perplexity': 6, 'n_iter': 15000})
    test_euc_list = []
    for x in test_all_umap[:-1]:
        test_euc_list.append(np.sqrt(np.sum((test_all_umap[-1] - x) ** 2)))
    test_euc = pd.DataFrame(zip(annotation_file.iloc[index, 0], test_euc_list), columns=['identifier', 'distance'])
    like = test_euc[test_euc.loc[:, 'distance'] == test_euc.loc[:, 'distance'].min()]

    test_euc.loc[len(test_euc.index)] = ['test unknown', 0]
    test_euc['component_0'] = test_all_umap[:, 0]
    test_euc['component_1'] = test_all_umap[:, 1]
    test_euc['component_2'] = test_all_umap[:, 2]
    test_euc.sort_values(by="distance", inplace=True, ascending=True)
    test_euc.to_csv(output_path + '/' + prefix + '_' + labels[pre_class] + '_unknown.csv', index=None)

    # Get distance figure
    test_known = test_euc.iloc[1:, :]
    test_unknown = test_euc.iloc[0, :]

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
        hovertext=test_known.loc[:, 'identifier']
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

    fig.write_html(output_path + '/' + prefix + '_' + labels[pre_class] + '.html')

    return test_euc.iloc[1:4, :]


def mkdir(output_path):
    folder = os.path.exists(output_path)
    if not folder:
        os.makedirs(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, help="Input fasta file")
    parser.add_argument("--embedding", type=str, help="Input embeddings")
    parser.add_argument("--output", type=str, help="Custom Output directory path")
    parser.add_argument("--prefix", type=str, help="Custom Output filename")

    args = parser.parse_args()

    output_path = args.output
    prefix = args.prefix

    ksb_predict(args.fasta, args.embedding, output_path, prefix)