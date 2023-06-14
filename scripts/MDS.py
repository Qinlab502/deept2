import torch
from torch.autograd import Variable
import numpy as np
import sklearn.covariance

class get_MDS:
    def __init__(self, model, num_classes, feature_list):
        self.model = model
        self.num_classes = num_classes
        self.feature_list = feature_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_class_mean(self, train_loader):

        model = self.model
        num_classes = self.num_classes
        feature_list = self.feature_list
        device = self.device

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
            data = data.to(device)
            data = Variable(data)
            output, out_features = model.feature_list(data)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            # equal_flag = pred.eq(target.cuda()).cpu()
            equal_flag = pred.eq(target.to(device)).cpu()
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
                        list_features[out_count][label] = torch.cat(
                            (list_features[out_count][label], out[i].view(1, -1)),
                            0)
                        out_count += 1
                num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
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
            # temp_precision = torch.from_numpy(temp_precision).float().cuda()
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

    def Mahalanobis_score(self, test_loader, outf, out_flag, sample_mean, precision, layer_index, magnitude):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''

        model = self.model
        num_classes = self.num_classes
        device = self.device
        
        model.eval()
        Mahalanobis = []

        if out_flag == True:
            temp_file_name = '%s/confidence_Ga%s_In.txt' % (outf, str(layer_index))
        else:
            temp_file_name = '%s/confidence_Ga%s_Out.txt' % (outf, str(layer_index))

        g = open(temp_file_name, 'w')

        for data, target in test_loader:

            # data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)
            data, target = Variable(data, requires_grad=True), Variable(target)

            out_features = model.intermediate_forward(data, layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)

            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            tempInputs = torch.add(data.data, gradient, alpha=-magnitude)

            noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

            for i in range(data.size(0)):
                g.write("{}\n".format(noise_gaussian_score[i]))
        g.close()

        return Mahalanobis
