---
layout: post
title: Training Models Across Multiple Machines - The Federated Learning Way
date:   2024-07-13 12:57:49 +0000
categories: jekyll update
excerpt: Notes on Distributed Training
mathjax: true
---

What if you could train a machine learning model on thousands of devices without ever collecting their data in one place? That's the core idea behind Federated Learning (FL). Instead of moving data to the model, we move the model to the data.

In this post, I'll walk through building a federated learning system from scratch using gRPC, Protocol Buffers, and a custom ResNet model. By the end, you'll see how these pieces fit together to create a privacy-preserving distributed training system.

<div class="imgcap">
<img src="/assets/fl/fl1.png" width="800" style="border: none;">
<figcaption>Figure 1: Starting Training on Two different machine</figcaption>
</div>

## How Federated Learning Works

The architecture is straightforward. There are two main players:

- **Leader (Server):** Maintains the global model and coordinates the training process
- **Learners (Clients):** Train on their local data and send back model updates

Here's the basic flow:

1. Learners register with the leader and receive their data split
2. Learners download the current global model
3. Each learner trains locally and computes gradients
4. Learners send gradients back to the leader
5. Leader aggregates gradients and updates the global model
6. Repeat until convergence

The beauty of this approach is that raw data never leaves the learner's device. Only model updates travel across the network.

## The Leader Service

The leader (`leader.py`) acts as the coordination hub. It initializes the global model, manages learner registrations, and aggregates incoming gradients. 

I chose gRPC for communication because it's efficient and handles complex data serialization well through Protocol Buffers. This matters when you're shuffling large models and datasets across the network.

## The Learner Service

Learners (`learner.py`) do the actual training work. Each learner:

- Syncs its local model with the global one
- Trains on its local dataset
- Computes gradients
- Sends gradients back to the leader

This keeps the global model improving while data stays distributed.

## Communication Layer: gRPC and Protocol Buffers

gRPC provides the backbone for leader-learner communication. Protocol Buffers define the message structure and service contracts, ensuring both sides speak the same language.

Let's look at the leader's protocol definition in `leader.proto`:

```protobuf
syntax = "proto3";

package leader;

service LeaderService {
    rpc RegisterLearner(LearnerInfo) returns (AckWithMetadata) {};
    rpc GetModel(Empty) returns (stream ModelChunk) {};
    rpc GetData(LearnerDataRequest) returns (stream DataChunk) {};
    rpc AccumulateGradients(GradientData) returns (Ack) {};
}
```

Each RPC serves a specific purpose:

**RegisterLearner** - When a learner joins, it sends its network address and gets back a unique ID plus metadata about the system.

**GetModel** - Learners request the latest global model. We stream it in chunks because models can be large.

**GetData** - The leader sends the learner its data split, again in chunks.

**AccumulateGradients** - After local training, learners send their computed gradients here for aggregation.

The message types handle the actual data:

```protobuf
message LearnerInfo {
    string network_addr = 1;
}

message AckWithMetadata {
    bool success = 1;
    string message = 2;
    int32 learner_id = 3;
    int32 max_learners = 4;
}

message ModelChunk {
    bytes chunk = 1;
}

message GradientData {
    bytes chunk = 1;
}
```

Using `bytes` for chunks gives us flexibility - we can serialize the data however we want before sending it over the wire.

## The Learner Protocol

On the learner side, `learner.proto` defines how learners receive training commands:

```protobuf
syntax = "proto3";

package learner;

service LearnerService {
    rpc StartTraining(Empty) returns (Ack) {};
    rpc SyncModelState(ModelState) returns (Ack) {};
}

message ModelState {
    bytes chunk = 1;
}

message Ack {
    bool success = 1;
    string message = 2;
}
```

**StartTraining** kicks off the local training process, while **SyncModelState** ensures the learner has the latest global model before training begins.

## The Model

For the model itself (`model.py`), I went with ResNet. Its residual connections help gradients flow through deep networks without vanishing, making it solid for image classification.

Training uses standard SGD with cross-entropy loss - nothing fancy, but effective.

## Data Handling

The `utils.py` file manages data loading and preprocessing. For something like CIFAR-10, it:

- Splits the dataset across learners
- Applies standard augmentations (normalization, resizing)
- Ensures each learner gets a unique data subset

Good data preprocessing is crucial for model generalization.

## The Complete Workflow

Let me trace through a complete training cycle:

1. A learner starts up and calls `RegisterLearner`, getting back its ID and finding out how many other learners exist
2. It calls `GetData` to fetch its training data split
3. It calls `GetModel` to download the current global model
4. Local training happens - the learner runs SGD on its data
5. It calls `AccumulateGradients` to send its computed gradients to the leader
6. The leader aggregates gradients from all learners and updates the global model
7. Repeat from step 3 until the model converges

<div class="imgcap">
<img src="/assets/fl/fl2.png" width="800" style="border: none;">
<figcaption>Figure 2: Accumlating Gradients from learner </figcaption>
</div>

## Federated Learning vs Distributed Training

People sometimes confuse federated learning with distributed training, but there's a key difference.

In distributed training, you typically gather all your data in one place, then split it across multiple machines to parallelize training. It's about speed - using multiple GPUs or nodes to train faster.

Federated learning keeps data where it lives. Each learner trains on its own data, and only model updates get shared. The raw data never moves. This makes federated learning inherently more private, though potentially slower since you're coordinating across potentially unreliable client devices.

## Wrapping Up

Federated learning offers a different way to think about model training. By keeping data decentralized and only sharing model updates, we can build systems that respect privacy while still leveraging data from many sources.

The system I've described here - with gRPC handling communication, Protocol Buffers defining the contracts, and a straightforward aggregation strategy - provides a solid foundation for experimenting with federated learning.

**[Code Available at Github](https://github.com/anshulsc/metal-FL)**

---