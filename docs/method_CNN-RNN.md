### 3. Method
Since we aim to characterize the high-order label correlation, we employ long short term memory (LSTM) neurons [15] as our recurrent neurons, which has been demonstrated to be a powerful model of long-term dependency.

#### 3.1. Long Short Term Memory Networks (LSTM)
RNN [15] is a class of neural network that maintains internal hidden states to model the dynamic temporal behaviour of sequences with arbitrary lengths through directed cyclic connections between its units. It can be considered as a hidden Markov model extension that employs nonlinear transition function and is capable of modeling long term temporal dependencies. LSTM extends RNN by adding three gates to an RNN neuron: a forget gate f to control whether to forget the current state; an input gate i to indicate if it should read the input; an output gate O to control whether to output the state. These gates enable LSTM to learn long-term dependency in a sequence, and make it is easier to optimize, because these gates help the input signal to effectively propagate through the recurrent hidden states \(r(t)\) without affecting the output. LSTM also effectively deals with the gradient vanishing/exploding issues that commonly appear during RNN training [26]. 

$$
\begin{aligned} 
x_{t} & =\delta\left(U_{r} \cdot r(t-1)+U_{w} w_{k}(t)\right) \\ 
i_{t} & =\delta\left(U_{i_{r}} r(t-1)+U_{i_{w}} w_{k}(t)\right) \\ 
f_{t} & =\delta\left(U_{f_{r}} r(t-1)+U_{f_{w}} w_{k}(t)\right) \\ 
o_{t} & =\delta\left(U_{o_{r}} r(t-1)+U_{o_{w}} w_{k}(t)\right) \\ 
r(t) & =f_{t} \odot r(t-1)+i_{t} \odot x_{t} \\ 
o(t) & =r(t) \odot o(t) 
\end{aligned}
$$
where \(\delta(.\) is an activation function, \(\odot\) is the product with gate value, and various W matrices are learned parameters. In our implementation, we employ rectified linear units (ReLU) as the activation function [4].

#### 3.2. Model
We propose a novel CNN-RNN framework for multi-label classification problem. The illustration of the CNN-RNN framework is shown in Fig. 4. It contains two parts: The CNN part extracts semantic representations from images; the RNN part models image/label relationship and label dependency.

We decompose a multi-label prediction as an ordered prediction path. For example, labels "zebra" and "elephant" can be decomposed as either ("zebra", "elephant") or ("elephant", "zebra"). The probability of a prediction path can be computed by the RNN network. The image, label, and recurrent representations are projected to the same low-dimensional space to model the image-text relationship as well as the label redundancy. The RNN model is employed as a compact yet powerful representation of the label co-occurrence dependency in this space. It takes the embedding of the predicted label at each time step and maintains a hidden state to model the label co-occurrence information. The a priori probability of a label given the previously predicted labels can be computed according to their dot products with the sum of the image and recurrent embeddings. The probability of a prediction path can be obtained as the product of the a-prior probability of each label given the previous labels in the prediction path.

A label k is represented as a one-hot vector \(e_{k} = [0, ... 0,1,0, ..., 0]\) , which is 1 at the k-th location, and 0 elsewhere. The label embedding can be obtained by multiplying the one-hot vector with a label embedding matrix \(U_{l}\) The k-th row of \(U_{l}\) is the label embedding of the label k 
$$
w_{k}=U_{l} . e_{k} . (2)
$$

The dimension of \(w_{k}\) is usually much smaller than the number of labels.

The recurrent layer takes the label embedding of the previously predicted label, and models the co-occurrence dependencies in its hidden recurrent states by learning nonlinear functions: 
$$
o(t)=h_{o}\left(r(t-1), w_{k}(t)\right), r(t)=h_{r}\left(r(t-1), w_{k}(t)\right)
$$
where \(r(t)\) and \(o(t)\) are the hidden states and outputs of the recurrent layer at the time step t, respectively, \(w_{k}(t)\) is the label embedding of the t-th label in the prediction path, and \(ho(.)\) , \(h_{r}(.\) are the non-linear RNN functions, which will be described in details in Sec. 3.1.

The output of the recurrent layer and the image representation are projected into the same low-dimensional space as the label embedding. 
$$
x_{t}=h\left(U_{o}^{x} o(t)+U_{I}^{x} I\right), (4)
$$
where \(U_{o}^{x}\) and \(U_{I}^{x}\) are the projection matrices for recurrent layer output and image representation, respectively. The number of columns of \(U_{o}^{x}\) and \(U_{I}^{x}\) are the same as the label embedding matrix \(U_{l}\) I is the convolutional neural network image representation. We will show in Sec 4.5 that the learned joint embedding effectively characterizes the relevance of images and labels.

Finally, the label scores can be computed by multiplying the transpose of \(U_{l}\) and \(x_{t}\) to compute the distances between \(x_{t}\) and each label embedding. 
$$
s(t)=U_{l}^{T} x_{t} . (5)
$$

The predicted label probability can be computed using softmax normalization on the scores.

#### 3.3. Inference
A prediction path is a sequence of labels \((l_{1}, l_{2}, l_{3}, \cdots, l_{N})\) , where the probability of each label \(l_{t}\) can be computed with the information of the image I and the previously predicted labels \(l_{1}, \cdots, l_{t-1}\) . The RNN model predicts multiple labels by finding the prediction path that maximizes the a priori probability. 
$$
\begin{aligned} 
l_{1}, \cdots, l_{k} & =arg max _{l_{1}, \cdots, l_{k}} P\left(l_{1}, \cdots, l_{k} | I\right) \\ 
& =arg max _{l_{1}, \cdots, l_{k}} P\left(l_{1} | I\right) × P\left(l_{2} | I, l_{1}\right) . \\ 
& \cdots P\left(l_{k} | I, l_{1}, \cdots, l_{k-1}\right) 
\end{aligned}
$$

Since the probability \(P(l_{k} | I, l_{1}, \cdots, l_{k-1})\) does not have Markov property, there is no optimal polynomial algorithm to find the optimal prediction path. We can employ the greedy approximation, which predicts label \(\hat{l}_{t} = arg max, P(l_{t} | I, l_{1}, \cdots, l_{t-1})\) at time step t and fix the label prediction \(\hat{l}_{t}\) at later predictions. However, the greedy algorithm is problematic because if the first predicted label is wrong, it is very likely that the whole sequence cannot be correctly predicted. Thus, we employ the beam search algorithm to find the top-ranked prediction path.

An example of the beam search algorithm can be found in Figure 5. Instead of greedily predicting the most probable label, the beam search algorithm finds the top-N most probable prediction paths as intermediate paths \(S(t)\) at each time step t 
$$
\mathcal{S}(t)=\left\{P_{1}(t), P_{2}(t), \cdots, P_{N}(t)\right\}
$$

At time step \(t+1\) , we add N most probable labels to each intermediate path \(P_{i}(t)\) to get a total of \(N ×N\) paths. The N prediction paths with highest probability among these paths constitute the intermediate paths for time step \(t+1\) . The prediction paths ending with the END sign are added to the candidate path set c . The termination condition of the beam search is that the probability of the current intermediate paths is smaller than that of all the candidate paths. It indicates that we cannot find any more candidate paths with greater probability.

#### 3.4. Training
Learning CNN-RNN models can be achieved by using the cross-entropy loss on the softmax normalization of score softmax \((s(t))\) and employing back-propagation through time algorithm. In order to avoid the gradient vanishing/exploding issues, we apply the rmsprop optimization algorithm [33]. Although it is possible to fine-tune the convolutional neural network in our architecture, we keep the convolutional neural network unchanged in our implementation for simplicity.

One important issue of training multi-label CNN-RNN models is to determine the orders of the labels. In the experiments of this paper, the label orders during training are determined according to their occurrence frequencies in the training data. More frequent labels appear earlier than the less frequent ones, which corresponds to the intuition that easier objects should be predicted first to help predict more difficult objects. We explored learning label orders by iteratively finding the easiest prediction ordering and order ensembles as proposed in [28] or simply using fixed random order, but they do not have notable effects on the performance. We also attempted to randomly permute the label orders in each mini-batch, but it makes the training very difficult to converge.