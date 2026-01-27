# Kubernetes

Kubernetes (K8s) is a container orchestration platform for automating deployment, scaling, and management.

## Core Components

- **Pod**: The smallest deployable unit, contains one or more containers.
- **Service**: An abstract way to expose an application running on Pods.
- **Deployment**: Manages the desired state of Pods and ReplicaSets.
- **Namespace**: Virtual clusters within a physical cluster.

## Architecture

Kubernetes uses a master-worker architecture:
- **Control Plane**: API Server, etcd, Scheduler, Controller Manager
- **Worker Nodes**: kubelet, kube-proxy, container runtime

Kubernetes can run Docker containers and manages their lifecycle across multiple nodes.
