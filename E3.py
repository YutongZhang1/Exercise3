"""
Dependencies required to run this script:

1. torch - for building and training neural network models.
2. torch.nn.functional - for applying neural network operations.
3. matplotlib.pyplot - for creating figures and visualizations.
4. random - for generating pseudo-random numbers for various uses.

Ensure these libraries are installed in your Python environment before execution.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import random


# Open the dataset file and read its contents
file_path = '/Users/zhangyutong/Desktop/animal_names_ch.txt' # NOTE: Please change the 'file_path' to the path where your file is located on your computer
with open(file_path, 'r') as file:
    animal_names = file.read()
    print("Original:", animal_names)

# Tokenize the dataset into individual animal names
animal_names = animal_names.split('\n')

# Normalize the animal names and remove the trailing "。"
normalized_animal_names = [name.strip().rstrip('。') for name in animal_names]

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(normalized_animal_names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['。'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# Define the block size for the model
block_size = 3 # Context length

# Function to build the dataset
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '。':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # Crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print("X shape:", X.shape, "Y shape:", Y.shape)
    return X, Y

# Split the dataset into train, dev, and test sets
random.seed(42)
random.shuffle(normalized_animal_names)
n1 = int(0.8 * len(normalized_animal_names))
n2 = int(0.9 * len(normalized_animal_names))

Xtr, Ytr = build_dataset(normalized_animal_names[:n1])
Xdev, Ydev = build_dataset(normalized_animal_names[n1:n2])
Xte, Yte = build_dataset(normalized_animal_names[n2:])

# Iterates through the first 20 elements of Xtr and Ytr, converting indices to strings using a mapping (itos), and prints each sequence alongside its label
for x,y in zip(Xtr[:20], Ytr[:20]):
  print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])

# Train the network
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # Note: kaiming init
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


# -----------------------------------------------------------------------------------------------
class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # Parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # Buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
    
  def __call__(self, x):
    # Calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0, 1)
      xmean = x.mean(dim, keepdim=True) # Batch mean
      xvar = x.var(dim, keepdim=True) # Batch variance
      xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # Normalize to unit variance
      # Update running mean and variance
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
    else:
      # Use running mean and variance for normalization
      xmean = self.running_mean
      xvar = self.running_var
      xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # Normalize

    self.out = self.gamma * xhat + self.beta
    return self.out
    
  def parameters(self):
    return [self.gamma, self.beta]

    # -----------------------------------------------------------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

# -----------------------------------------------------------------------------------------------
# Basic embedding layer: maps indices to dense vectors.
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]

# -----------------------------------------------------------------------------------------------
class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    if x.dim() == 3:  # If x has 3 dimensions: [B, T, C]
        B, T, C = x.shape
    elif x.dim() == 2:  # If x has 2 dimensions: [B, C] or [T, C]
        x = x.unsqueeze(1)  # Add a singleton dimension to make it [B, T=1, C] or [T=1, C]
        B, T, C = x.shape
    else:
        raise ValueError("Input x must have 2 or 3 dimensions")
    
    remainder = T % self.n
    if remainder != 0:
            # Calculate padding size to make T divisible by n
            padding_size = self.n - remainder
            # Pad the sequence on the right for each batch
            x = F.pad(x, (0, 0, 0, padding_size), "constant", 0)
            # Update T to reflect the new padded size
            T += padding_size
    x = x.view(B, T//self.n, C*self.n)
    self.out = x
    return self.out
  
  def parameters(self):
    return []
  # -----------------------------------------------------------------------------------------------
class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # Get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]

torch.manual_seed(42) # Seed rng for reproducibility

# Hierarchical network
n_embd = 26 # The dimensionality of the character embedding vectors
n_hidden = 128 # The number of neurons in the hidden layer of the MLP
model = Sequential([
  Embedding(vocab_size, n_embd),
  FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
])

# Function to build and return the model with specified embedding size
def build_model(vocab_size, n_embd, n_hidden, output_size):
    return Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), 
        Linear(n_embd * 2, n_hidden, bias=False), 
        BatchNorm1d(n_hidden), 
        Tanh(),
        FlattenConsecutive(2), 
        Linear(n_hidden*2, n_hidden, bias=False), 
        BatchNorm1d(n_hidden), 
        Tanh(),
        FlattenConsecutive(2), 
        Linear(n_hidden*2, n_hidden, bias=False), 
        BatchNorm1d(n_hidden), 
        Tanh(),
        Linear(n_hidden, output_size),
    ])

# Main experiment loop
embedding_sizes = [2, 16, 64, 128]
results = []

best_accuracy = 0.0
best_embedding_size = None

for n_embd in embedding_sizes:
    model = build_model(vocab_size, n_embd, n_hidden=128, output_size=vocab_size) 
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True

    # Function to train the model
    def train_model(model, X, Y, epochs=10, lr=0.01):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits.squeeze(1), Y)
            loss.backward()
            optimizer.step()
        print(f"Finished training with embedding size {n_embd}. Loss: {loss.item()}")

    # Function to evaluate the model
    def evaluate_model(model, X, Y):
        logits = model(X)
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == Y).float().mean()
        return accuracy.item()

    # Train and evaluate the model
    train_model(model, Xtr, Ytr, epochs=10, lr=0.01)
    accuracy = evaluate_model(model, Xte, Yte)
    print(f"Embedding Size: {n_embd}, Test Accuracy: {accuracy:.4f}")
    results.append((n_embd, accuracy))

    # Track the best performing model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_embedding_size = n_embd

# Display the best embedding size and its accuracy
print(f"The best embedding size is {best_embedding_size} with an accuracy of {best_accuracy:.4f}.")

# Visualization of results
embedding_sizes, accuracies = zip(*results)
plt.plot(embedding_sizes, accuracies, marker='o')
plt.title('Model Accuracy vs. Embedding Size')
plt.xlabel('Embedding Size')
plt.ylabel('Accuracy')
plt.show()

# Parameter init
with torch.no_grad():
  model.layers[-1].weight *= 0.1 # Last layer make less confident

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # Number of parameters in total
for p in parameters:
  p.requires_grad = True

# Same optimization as last time
max_steps = 5000
batch_size = 16
lossi = []

for i in range(max_steps):
  
    # Minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix] # Batch X, Y

    # Forward pass
    logits = model(Xb)
    logits = torch.squeeze(logits, dim=1)

    # Compute the loss
    loss = F.cross_entropy(logits, Yb)
  
    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # Update: simple SGD
    lr = 0.01 if i < 2500 else 0.001
    for p in parameters:
        p.data += -lr * p.grad

    # Track stats
    if i % 250 == 0: # Print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

# Visualization of the loss
plt.title('Visualization of the loss')
plt.plot(torch.tensor(lossi).view(-1, 500).mean(1))
plt.show()

# Example instantiation of the Embedding class
embedding_instance = Embedding(num_embeddings=len(chars) + 1, embedding_dim=2)  # Assuming 2D embeddings for simplicity

# Access the embedding matrix
C = embedding_instance.weight.data.numpy()  # Assuming the embeddings are 2D

# Visualization of the semantic distribution
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0], C[:, 1], s=200)

for i, char in enumerate(chars):
    plt.text(C[i, 0].item(), C[i, 1].item(), char, ha="center", va="center", color='white')

plt.grid(True, which='minor')  # Enable a grid for better visualization
plt.title('Embedding Visualization with Chinese Characters')
plt.xlabel('Dimension 0')  # Optionally, adjust or localize labels
plt.ylabel('Dimension 1')
plt.show()

# Put layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
  layer.training = False

num_correct = 0
num_samples = 0

with torch.no_grad():
    for i in range(0, Xte.size(0), batch_size):
        x_batch = Xte[i:i+batch_size]
        y_batch = Yte[i:i+batch_size]

        # Forward pass
        logits = model(x_batch)
        logits = logits.squeeze(1)  # Adjust logits if necessary

        # Prediction
        _, predicted = torch.max(logits, 1)

        # Update counts
        num_correct += (predicted == y_batch).sum().item()
        num_samples += y_batch.size(0)

# Calculate and print accuracy
accuracy = 100 * num_correct / num_samples
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

# Evaluate the loss
@torch.no_grad() # This decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  logits = model(x)
  logits = logits.squeeze(1)
  loss = torch.nn.functional.cross_entropy(input=logits, target=y)
  print(f"{split} Loss:", loss.item())

split_loss('train')
split_loss('val')

# Sample from the model
for _ in range(20):
    
    out = []
    context = [0] * block_size # Initialize with all ...
    while True:
      # Forward pass the neural net
      logits = model(torch.tensor([context]))
      probs = torch.softmax(logits, dim=-1)
      # Sample from the distribution
      ix = torch.multinomial(probs.view(-1), num_samples=1).item()
      # Shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # If we sample the special '。' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # Decode and print the generated word
