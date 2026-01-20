#!/usr/bin/env python3
"""Create semantically connected test data."""

import asyncio
import random
import uuid

from engram.storage.neo4j_client import Neo4jClient


# Concepts with their actual semantic relationships
KNOWLEDGE_GRAPH = {
    # ===================
    # PHYSICS
    # ===================

    # Classical Mechanics
    "newton_laws": {
        "type": "principle", "domain": "physics",
        "connects": ["force", "mass", "acceleration", "momentum", "inertia", "classical_mechanics"],
        "desc": "Three fundamental laws describing motion and forces"
    },
    "force": {
        "type": "concept", "domain": "physics",
        "connects": ["newton_laws", "mass", "acceleration", "gravity", "friction", "momentum", "vector"],
        "desc": "Interaction that changes object's motion"
    },
    "momentum": {
        "type": "concept", "domain": "physics",
        "connects": ["mass", "velocity", "newton_laws", "collision", "conservation", "impulse"],
        "desc": "Product of mass and velocity, conserved in closed systems"
    },
    "energy": {
        "type": "concept", "domain": "physics",
        "connects": ["kinetic_energy", "potential_energy", "work", "conservation", "thermodynamics", "joule"],
        "desc": "Capacity to do work, conserved quantity"
    },
    "kinetic_energy": {
        "type": "concept", "domain": "physics",
        "connects": ["energy", "velocity", "mass", "work", "momentum"],
        "desc": "Energy of motion"
    },
    "potential_energy": {
        "type": "concept", "domain": "physics",
        "connects": ["energy", "gravity", "height", "spring", "work", "field"],
        "desc": "Stored energy due to position or configuration"
    },
    "gravity": {
        "type": "force", "domain": "physics",
        "connects": ["mass", "newton_laws", "gravitational_constant", "general_relativity", "orbit", "weight"],
        "desc": "Attractive force between masses"
    },
    "classical_mechanics": {
        "type": "field", "domain": "physics",
        "connects": ["newton_laws", "lagrangian", "hamiltonian", "momentum", "energy", "kinematics"],
        "desc": "Study of motion of macroscopic objects"
    },
    "lagrangian": {
        "type": "formalism", "domain": "physics",
        "connects": ["classical_mechanics", "hamiltonian", "action", "variational_principle", "euler_lagrange"],
        "desc": "Function describing system dynamics via kinetic minus potential energy"
    },
    "hamiltonian": {
        "type": "formalism", "domain": "physics",
        "connects": ["classical_mechanics", "lagrangian", "energy", "phase_space", "quantum_mechanics"],
        "desc": "Function representing total energy in phase space"
    },

    # Quantum Mechanics
    "quantum_mechanics": {
        "type": "field", "domain": "physics",
        "connects": ["wave_function", "schrodinger_equation", "hamiltonian", "superposition", "entanglement",
                     "heisenberg_uncertainty", "planck_constant", "quantum_field_theory"],
        "desc": "Physics of atomic and subatomic particles"
    },
    "wave_function": {
        "type": "concept", "domain": "physics",
        "connects": ["quantum_mechanics", "schrodinger_equation", "probability", "superposition", "collapse"],
        "desc": "Mathematical description of quantum state"
    },
    "schrodinger_equation": {
        "type": "equation", "domain": "physics",
        "connects": ["quantum_mechanics", "wave_function", "hamiltonian", "eigenvalue", "time_evolution"],
        "desc": "Fundamental equation describing quantum state evolution"
    },
    "superposition": {
        "type": "principle", "domain": "physics",
        "connects": ["quantum_mechanics", "wave_function", "interference", "qubit", "measurement"],
        "desc": "Quantum state existing in multiple states simultaneously"
    },
    "entanglement": {
        "type": "phenomenon", "domain": "physics",
        "connects": ["quantum_mechanics", "superposition", "bell_inequality", "qubit", "quantum_computing"],
        "desc": "Correlated quantum states across distance"
    },
    "heisenberg_uncertainty": {
        "type": "principle", "domain": "physics",
        "connects": ["quantum_mechanics", "position", "momentum", "planck_constant", "wave_function"],
        "desc": "Fundamental limit on simultaneous measurement precision"
    },
    "planck_constant": {
        "type": "constant", "domain": "physics",
        "connects": ["quantum_mechanics", "energy", "frequency", "photon", "heisenberg_uncertainty"],
        "desc": "Fundamental constant relating energy and frequency"
    },
    "quantum_field_theory": {
        "type": "field", "domain": "physics",
        "connects": ["quantum_mechanics", "special_relativity", "particle_physics", "feynman_diagram", "qed"],
        "desc": "Framework combining quantum mechanics and special relativity"
    },

    # Thermodynamics
    "thermodynamics": {
        "type": "field", "domain": "physics",
        "connects": ["entropy", "temperature", "heat", "energy", "statistical_mechanics", "first_law", "second_law"],
        "desc": "Study of heat, work, and energy"
    },
    "entropy": {
        "type": "concept", "domain": "physics",
        "connects": ["thermodynamics", "second_law", "disorder", "statistical_mechanics", "information_theory"],
        "desc": "Measure of disorder or randomness"
    },
    "temperature": {
        "type": "concept", "domain": "physics",
        "connects": ["thermodynamics", "heat", "kinetic_energy", "boltzmann_constant", "absolute_zero"],
        "desc": "Measure of average kinetic energy of particles"
    },
    "heat": {
        "type": "concept", "domain": "physics",
        "connects": ["thermodynamics", "temperature", "energy", "conduction", "convection", "radiation"],
        "desc": "Energy transfer due to temperature difference"
    },
    "second_law": {
        "type": "law", "domain": "physics",
        "connects": ["thermodynamics", "entropy", "irreversibility", "heat_engine", "carnot"],
        "desc": "Entropy of isolated system never decreases"
    },
    "statistical_mechanics": {
        "type": "field", "domain": "physics",
        "connects": ["thermodynamics", "entropy", "boltzmann", "partition_function", "microstate"],
        "desc": "Statistical approach to thermodynamics"
    },
    "boltzmann_constant": {
        "type": "constant", "domain": "physics",
        "connects": ["statistical_mechanics", "temperature", "entropy", "ideal_gas", "kinetic_theory"],
        "desc": "Relates temperature to energy at molecular level"
    },

    # Electromagnetism
    "electromagnetism": {
        "type": "field", "domain": "physics",
        "connects": ["maxwell_equations", "electric_field", "magnetic_field", "electromagnetic_wave", "charge"],
        "desc": "Study of electric and magnetic phenomena"
    },
    "maxwell_equations": {
        "type": "equations", "domain": "physics",
        "connects": ["electromagnetism", "electric_field", "magnetic_field", "electromagnetic_wave", "gauss_law"],
        "desc": "Four equations describing electromagnetism"
    },
    "electric_field": {
        "type": "concept", "domain": "physics",
        "connects": ["electromagnetism", "charge", "coulomb_law", "potential", "maxwell_equations"],
        "desc": "Field created by electric charges"
    },
    "magnetic_field": {
        "type": "concept", "domain": "physics",
        "connects": ["electromagnetism", "current", "lorentz_force", "inductance", "maxwell_equations"],
        "desc": "Field created by moving charges"
    },
    "electromagnetic_wave": {
        "type": "phenomenon", "domain": "physics",
        "connects": ["electromagnetism", "maxwell_equations", "light", "photon", "wavelength", "frequency"],
        "desc": "Propagating oscillation of electric and magnetic fields"
    },
    "photon": {
        "type": "particle", "domain": "physics",
        "connects": ["electromagnetic_wave", "light", "quantum_mechanics", "energy", "planck_constant", "qed"],
        "desc": "Quantum of electromagnetic radiation"
    },

    # Relativity
    "special_relativity": {
        "type": "theory", "domain": "physics",
        "connects": ["speed_of_light", "time_dilation", "length_contraction", "lorentz_transformation", "mass_energy"],
        "desc": "Physics at velocities approaching light speed"
    },
    "general_relativity": {
        "type": "theory", "domain": "physics",
        "connects": ["gravity", "spacetime", "curvature", "tensor", "black_hole", "gravitational_wave"],
        "desc": "Geometric theory of gravitation"
    },
    "spacetime": {
        "type": "concept", "domain": "physics",
        "connects": ["special_relativity", "general_relativity", "four_vector", "metric", "curvature"],
        "desc": "Four-dimensional continuum of space and time"
    },
    "speed_of_light": {
        "type": "constant", "domain": "physics",
        "connects": ["special_relativity", "electromagnetic_wave", "mass_energy", "causality"],
        "desc": "Maximum speed of information transfer"
    },
    "mass_energy": {
        "type": "principle", "domain": "physics",
        "connects": ["special_relativity", "energy", "mass", "speed_of_light", "nuclear"],
        "desc": "Equivalence of mass and energy (E=mc²)"
    },
    "black_hole": {
        "type": "object", "domain": "physics",
        "connects": ["general_relativity", "spacetime", "singularity", "event_horizon", "hawking_radiation"],
        "desc": "Region where gravity prevents light escape"
    },

    # Particle Physics
    "particle_physics": {
        "type": "field", "domain": "physics",
        "connects": ["standard_model", "quark", "lepton", "boson", "quantum_field_theory", "collider"],
        "desc": "Study of fundamental particles and forces"
    },
    "standard_model": {
        "type": "theory", "domain": "physics",
        "connects": ["particle_physics", "quark", "lepton", "boson", "higgs", "strong_force", "weak_force"],
        "desc": "Theory of fundamental particles and interactions"
    },
    "quark": {
        "type": "particle", "domain": "physics",
        "connects": ["standard_model", "proton", "neutron", "strong_force", "color_charge", "gluon"],
        "desc": "Fundamental particle making up hadrons"
    },
    "electron": {
        "type": "particle", "domain": "physics",
        "connects": ["lepton", "charge", "atom", "qed", "spin", "wave_function"],
        "desc": "Fundamental particle with negative charge"
    },
    "higgs": {
        "type": "particle", "domain": "physics",
        "connects": ["standard_model", "mass", "higgs_field", "symmetry_breaking", "lhc"],
        "desc": "Particle associated with mass generation"
    },

    # ===================
    # MATHEMATICS
    # ===================

    # Calculus
    "calculus": {
        "type": "field", "domain": "math",
        "connects": ["derivative", "integral", "limit", "continuity", "function", "differential_equation"],
        "desc": "Study of continuous change"
    },
    "derivative": {
        "type": "concept", "domain": "math",
        "connects": ["calculus", "limit", "rate_of_change", "gradient", "differentiation", "chain_rule"],
        "desc": "Rate of change of a function"
    },
    "integral": {
        "type": "concept", "domain": "math",
        "connects": ["calculus", "area", "antiderivative", "riemann_sum", "fundamental_theorem"],
        "desc": "Accumulation or area under curve"
    },
    "limit": {
        "type": "concept", "domain": "math",
        "connects": ["calculus", "continuity", "derivative", "epsilon_delta", "convergence"],
        "desc": "Value approached as input approaches some value"
    },
    "differential_equation": {
        "type": "concept", "domain": "math",
        "connects": ["calculus", "derivative", "ode", "pde", "initial_value", "boundary_value"],
        "desc": "Equation involving derivatives"
    },
    "gradient": {
        "type": "concept", "domain": "math",
        "connects": ["derivative", "vector", "multivariable_calculus", "directional_derivative", "optimization"],
        "desc": "Vector of partial derivatives"
    },
    "taylor_series": {
        "type": "concept", "domain": "math",
        "connects": ["calculus", "derivative", "polynomial", "approximation", "convergence", "analytic"],
        "desc": "Polynomial approximation of functions"
    },

    # Linear Algebra
    "linear_algebra": {
        "type": "field", "domain": "math",
        "connects": ["matrix", "vector", "eigenvalue", "linear_transformation", "vector_space", "determinant"],
        "desc": "Study of vectors, matrices, and linear transformations"
    },
    "matrix": {
        "type": "concept", "domain": "math",
        "connects": ["linear_algebra", "vector", "determinant", "inverse", "multiplication", "rank"],
        "desc": "Rectangular array of numbers"
    },
    "vector": {
        "type": "concept", "domain": "math",
        "connects": ["linear_algebra", "matrix", "dot_product", "cross_product", "magnitude", "direction"],
        "desc": "Quantity with magnitude and direction"
    },
    "eigenvalue": {
        "type": "concept", "domain": "math",
        "connects": ["linear_algebra", "matrix", "eigenvector", "determinant", "characteristic_polynomial", "pca"],
        "desc": "Scalar factor in eigenvector equation"
    },
    "eigenvector": {
        "type": "concept", "domain": "math",
        "connects": ["eigenvalue", "matrix", "linear_transformation", "diagonalization"],
        "desc": "Vector unchanged in direction by linear transformation"
    },
    "determinant": {
        "type": "concept", "domain": "math",
        "connects": ["matrix", "linear_algebra", "inverse", "eigenvalue", "volume", "singular"],
        "desc": "Scalar encoding matrix properties"
    },
    "vector_space": {
        "type": "concept", "domain": "math",
        "connects": ["linear_algebra", "vector", "basis", "dimension", "subspace", "span"],
        "desc": "Collection of vectors with addition and scalar multiplication"
    },
    "linear_transformation": {
        "type": "concept", "domain": "math",
        "connects": ["linear_algebra", "matrix", "kernel", "image", "isomorphism", "rotation"],
        "desc": "Function preserving vector addition and scalar multiplication"
    },

    # Probability and Statistics
    "probability": {
        "type": "field", "domain": "math",
        "connects": ["random_variable", "distribution", "expectation", "variance", "bayes_theorem", "conditional"],
        "desc": "Study of randomness and uncertainty"
    },
    "random_variable": {
        "type": "concept", "domain": "math",
        "connects": ["probability", "distribution", "expectation", "variance", "sample_space"],
        "desc": "Variable whose value depends on random outcomes"
    },
    "distribution": {
        "type": "concept", "domain": "math",
        "connects": ["probability", "random_variable", "normal", "binomial", "poisson", "pdf", "cdf"],
        "desc": "Function describing probability of outcomes"
    },
    "normal_distribution": {
        "type": "concept", "domain": "math",
        "connects": ["distribution", "mean", "standard_deviation", "gaussian", "central_limit_theorem"],
        "desc": "Bell-shaped probability distribution"
    },
    "bayes_theorem": {
        "type": "theorem", "domain": "math",
        "connects": ["probability", "conditional", "prior", "posterior", "likelihood", "bayesian"],
        "desc": "Relates conditional probabilities"
    },
    "expectation": {
        "type": "concept", "domain": "math",
        "connects": ["probability", "random_variable", "mean", "variance", "moment"],
        "desc": "Average value of random variable"
    },
    "variance": {
        "type": "concept", "domain": "math",
        "connects": ["probability", "expectation", "standard_deviation", "spread", "covariance"],
        "desc": "Measure of spread around mean"
    },
    "hypothesis_testing": {
        "type": "method", "domain": "math",
        "connects": ["statistics", "p_value", "null_hypothesis", "significance", "confidence_interval"],
        "desc": "Statistical method for testing claims"
    },
    "regression": {
        "type": "method", "domain": "math",
        "connects": ["statistics", "linear_regression", "correlation", "prediction", "least_squares"],
        "desc": "Modeling relationship between variables"
    },

    # Abstract Algebra
    "abstract_algebra": {
        "type": "field", "domain": "math",
        "connects": ["group", "ring", "field_algebra", "homomorphism", "isomorphism", "symmetry"],
        "desc": "Study of algebraic structures"
    },
    "group": {
        "type": "structure", "domain": "math",
        "connects": ["abstract_algebra", "identity", "inverse", "associativity", "symmetry", "subgroup"],
        "desc": "Set with operation satisfying group axioms"
    },
    "ring": {
        "type": "structure", "domain": "math",
        "connects": ["abstract_algebra", "group", "addition", "multiplication", "ideal", "polynomial_ring"],
        "desc": "Set with two operations (addition and multiplication)"
    },
    "field_algebra": {
        "type": "structure", "domain": "math",
        "connects": ["abstract_algebra", "ring", "division", "real_numbers", "complex_numbers", "galois"],
        "desc": "Ring where every nonzero element has multiplicative inverse"
    },
    "homomorphism": {
        "type": "concept", "domain": "math",
        "connects": ["abstract_algebra", "group", "ring", "kernel", "image", "structure_preserving"],
        "desc": "Structure-preserving map between algebraic structures"
    },

    # Topology
    "topology": {
        "type": "field", "domain": "math",
        "connects": ["open_set", "continuous", "homeomorphism", "manifold", "compactness", "connectedness"],
        "desc": "Study of properties preserved under continuous deformation"
    },
    "manifold": {
        "type": "concept", "domain": "math",
        "connects": ["topology", "differential_geometry", "dimension", "chart", "atlas", "tangent_space"],
        "desc": "Space locally resembling Euclidean space"
    },
    "homeomorphism": {
        "type": "concept", "domain": "math",
        "connects": ["topology", "continuous", "bijection", "invariant", "topological_equivalence"],
        "desc": "Continuous bijection with continuous inverse"
    },
    "compactness": {
        "type": "property", "domain": "math",
        "connects": ["topology", "closed", "bounded", "cover", "finite_subcover"],
        "desc": "Topological property generalizing closed and bounded"
    },

    # Number Theory
    "number_theory": {
        "type": "field", "domain": "math",
        "connects": ["prime", "divisibility", "modular_arithmetic", "diophantine", "cryptography"],
        "desc": "Study of integers and their properties"
    },
    "prime": {
        "type": "concept", "domain": "math",
        "connects": ["number_theory", "divisibility", "factorization", "prime_factorization", "rsa"],
        "desc": "Integer greater than 1 with only 1 and itself as factors"
    },
    "modular_arithmetic": {
        "type": "concept", "domain": "math",
        "connects": ["number_theory", "remainder", "congruence", "cryptography", "rsa"],
        "desc": "Arithmetic with wraparound at modulus"
    },

    # Analysis
    "real_analysis": {
        "type": "field", "domain": "math",
        "connects": ["limit", "continuity", "convergence", "metric_space", "measure_theory", "lebesgue"],
        "desc": "Rigorous study of real numbers and functions"
    },
    "complex_analysis": {
        "type": "field", "domain": "math",
        "connects": ["complex_numbers", "analytic_function", "contour_integral", "residue", "cauchy"],
        "desc": "Study of functions of complex variables"
    },
    "measure_theory": {
        "type": "field", "domain": "math",
        "connects": ["real_analysis", "integration", "lebesgue", "sigma_algebra", "probability"],
        "desc": "Generalization of length, area, volume"
    },

    # Discrete Mathematics
    "discrete_math": {
        "type": "field", "domain": "math",
        "connects": ["combinatorics", "graph_theory", "logic", "set_theory", "algorithms"],
        "desc": "Study of discrete mathematical structures"
    },
    "graph_theory": {
        "type": "field", "domain": "math",
        "connects": ["discrete_math", "vertex", "edge", "path", "tree", "network", "algorithms"],
        "desc": "Study of graphs as mathematical structures"
    },
    "combinatorics": {
        "type": "field", "domain": "math",
        "connects": ["discrete_math", "counting", "permutation", "combination", "generating_function"],
        "desc": "Study of counting and arrangement"
    },
    "set_theory": {
        "type": "field", "domain": "math",
        "connects": ["discrete_math", "element", "subset", "union", "intersection", "cardinality"],
        "desc": "Study of collections of objects"
    },
    "logic": {
        "type": "field", "domain": "math",
        "connects": ["discrete_math", "proposition", "predicate", "proof", "boolean", "inference"],
        "desc": "Study of formal reasoning"
    },

    # Numerical Methods
    "numerical_analysis": {
        "type": "field", "domain": "math",
        "connects": ["approximation", "error_analysis", "interpolation", "numerical_integration", "root_finding"],
        "desc": "Study of algorithms for numerical computation"
    },
    "interpolation": {
        "type": "method", "domain": "math",
        "connects": ["numerical_analysis", "polynomial", "spline", "data_fitting", "approximation"],
        "desc": "Estimating values between known data points"
    },
    "numerical_integration": {
        "type": "method", "domain": "math",
        "connects": ["numerical_analysis", "integral", "quadrature", "simpson", "trapezoidal"],
        "desc": "Numerical approximation of integrals"
    },

    # Math super nodes
    "proof": {
        "type": "concept", "domain": "math",
        "connects": ["theorem", "lemma", "logic", "induction", "contradiction", "direct"],
        "desc": "Logical argument establishing truth"
    },
    "theorem": {
        "type": "concept", "domain": "math",
        "connects": ["proof", "lemma", "corollary", "conjecture", "axiom"],
        "desc": "Statement proven from axioms"
    },
    "function": {
        "type": "concept", "domain": "math",
        "connects": ["domain", "range", "mapping", "calculus", "linear_transformation", "continuous"],
        "desc": "Relation assigning exactly one output to each input"
    },

    # Physics super nodes
    "conservation": {
        "type": "principle", "domain": "physics",
        "connects": ["energy", "momentum", "charge", "symmetry", "noether_theorem"],
        "desc": "Quantity remaining constant in closed system"
    },
    "symmetry": {
        "type": "concept", "domain": "physics",
        "connects": ["conservation", "group", "invariance", "noether_theorem", "gauge"],
        "desc": "Invariance under transformation"
    },
    "field": {
        "type": "concept", "domain": "physics",
        "connects": ["electric_field", "magnetic_field", "gravitational", "quantum_field_theory", "scalar", "vector"],
        "desc": "Physical quantity defined at every point in space"
    },

    # ===================
    # ORIGINAL TECH DOMAINS
    # ===================

    # Machine Learning core
    "neural_network": {
        "type": "tool", "domain": "ml",
        "connects": ["deep_learning", "backpropagation", "gradient_descent", "activation_function",
                     "layer", "weights", "bias", "loss_function", "optimizer", "training"],
        "desc": "Computational model with interconnected nodes that learns patterns"
    },
    "deep_learning": {
        "type": "tool", "domain": "ml",
        "connects": ["neural_network", "cnn", "rnn", "transformer", "gpu", "tensorflow", "pytorch"],
        "desc": "Neural networks with many layers for complex pattern recognition"
    },
    "transformer": {
        "type": "architecture", "domain": "ml",
        "connects": ["attention", "self_attention", "bert", "gpt", "positional_encoding",
                     "encoder", "decoder", "embedding", "tokenizer"],
        "desc": "Attention-based architecture for sequence modeling"
    },
    "attention": {
        "type": "mechanism", "domain": "ml",
        "connects": ["transformer", "self_attention", "query", "key", "value", "softmax", "weights"],
        "desc": "Mechanism to focus on relevant parts of input"
    },
    "backpropagation": {
        "type": "algorithm", "domain": "ml",
        "connects": ["neural_network", "gradient_descent", "chain_rule", "loss_function", "weights"],
        "desc": "Algorithm for computing gradients through the network"
    },
    "gradient_descent": {
        "type": "algorithm", "domain": "ml",
        "connects": ["backpropagation", "learning_rate", "optimizer", "adam", "sgd", "loss_function"],
        "desc": "Optimization algorithm that minimizes loss by following gradients"
    },
    "cnn": {
        "type": "architecture", "domain": "ml",
        "connects": ["convolution", "pooling", "image_classification", "deep_learning", "resnet", "filter"],
        "desc": "Convolutional Neural Network for image processing"
    },
    "rnn": {
        "type": "architecture", "domain": "ml",
        "connects": ["lstm", "gru", "sequence", "hidden_state", "time_series", "nlp"],
        "desc": "Recurrent Neural Network for sequential data"
    },
    "lstm": {
        "type": "architecture", "domain": "ml",
        "connects": ["rnn", "gru", "forget_gate", "memory_cell", "sequence", "vanishing_gradient"],
        "desc": "Long Short-Term Memory network with gating mechanism"
    },
    "bert": {
        "type": "model", "domain": "ml",
        "connects": ["transformer", "nlp", "pretraining", "fine_tuning", "masked_language_model", "embedding"],
        "desc": "Bidirectional transformer for NLP tasks"
    },
    "gpt": {
        "type": "model", "domain": "ml",
        "connects": ["transformer", "language_model", "text_generation", "autoregressive", "decoder"],
        "desc": "Generative Pre-trained Transformer for text generation"
    },
    "embedding": {
        "type": "concept", "domain": "ml",
        "connects": ["vector", "word2vec", "transformer", "bert", "semantic", "dimension"],
        "desc": "Dense vector representation of discrete items"
    },
    "loss_function": {
        "type": "concept", "domain": "ml",
        "connects": ["neural_network", "cross_entropy", "mse", "gradient_descent", "optimization"],
        "desc": "Function that measures prediction error"
    },
    "optimizer": {
        "type": "tool", "domain": "ml",
        "connects": ["adam", "sgd", "gradient_descent", "learning_rate", "momentum"],
        "desc": "Algorithm that updates model parameters"
    },
    "adam": {
        "type": "algorithm", "domain": "ml",
        "connects": ["optimizer", "gradient_descent", "momentum", "learning_rate", "adaptive"],
        "desc": "Adaptive Moment Estimation optimizer"
    },
    "overfitting": {
        "type": "problem", "domain": "ml",
        "connects": ["regularization", "dropout", "validation", "training", "generalization"],
        "desc": "Model memorizes training data instead of learning patterns"
    },
    "dropout": {
        "type": "technique", "domain": "ml",
        "connects": ["regularization", "overfitting", "neural_network", "training"],
        "desc": "Randomly drops neurons during training to prevent overfitting"
    },

    # Databases
    "postgresql": {
        "type": "database", "domain": "db",
        "connects": ["sql", "relational", "acid", "index", "query", "transaction", "table"],
        "desc": "Open-source relational database system"
    },
    "mongodb": {
        "type": "database", "domain": "db",
        "connects": ["nosql", "document", "json", "schema_less", "replica_set", "sharding"],
        "desc": "Document-oriented NoSQL database"
    },
    "redis": {
        "type": "database", "domain": "db",
        "connects": ["cache", "key_value", "in_memory", "pub_sub", "session", "queue"],
        "desc": "In-memory data structure store"
    },
    "neo4j": {
        "type": "database", "domain": "db",
        "connects": ["graph", "cypher", "node", "relationship", "traversal", "pattern_matching"],
        "desc": "Native graph database"
    },
    "elasticsearch": {
        "type": "database", "domain": "db",
        "connects": ["search", "full_text", "index", "lucene", "aggregation", "kibana"],
        "desc": "Distributed search and analytics engine"
    },
    "sql": {
        "type": "language", "domain": "db",
        "connects": ["postgresql", "mysql", "query", "select", "join", "where", "index"],
        "desc": "Structured Query Language for relational databases"
    },
    "index": {
        "type": "concept", "domain": "db",
        "connects": ["query", "performance", "btree", "hash", "postgresql", "elasticsearch"],
        "desc": "Data structure for fast lookups"
    },
    "transaction": {
        "type": "concept", "domain": "db",
        "connects": ["acid", "commit", "rollback", "isolation", "postgresql", "consistency"],
        "desc": "Atomic unit of database work"
    },
    "acid": {
        "type": "concept", "domain": "db",
        "connects": ["transaction", "atomicity", "consistency", "isolation", "durability", "postgresql"],
        "desc": "Database transaction properties"
    },
    "sharding": {
        "type": "technique", "domain": "db",
        "connects": ["horizontal_scaling", "partition", "distributed", "mongodb", "cassandra"],
        "desc": "Distributing data across multiple servers"
    },
    "replication": {
        "type": "technique", "domain": "db",
        "connects": ["high_availability", "replica", "primary", "secondary", "failover"],
        "desc": "Copying data to multiple nodes"
    },

    # DevOps & Infrastructure
    "docker": {
        "type": "tool", "domain": "devops",
        "connects": ["container", "image", "dockerfile", "kubernetes", "registry", "compose"],
        "desc": "Platform for containerizing applications"
    },
    "kubernetes": {
        "type": "tool", "domain": "devops",
        "connects": ["docker", "container", "pod", "deployment", "service", "helm", "orchestration"],
        "desc": "Container orchestration platform"
    },
    "container": {
        "type": "concept", "domain": "devops",
        "connects": ["docker", "image", "isolation", "process", "namespace", "cgroup"],
        "desc": "Isolated environment for running applications"
    },
    "pod": {
        "type": "concept", "domain": "devops",
        "connects": ["kubernetes", "container", "deployment", "service", "node"],
        "desc": "Smallest deployable unit in Kubernetes"
    },
    "helm": {
        "type": "tool", "domain": "devops",
        "connects": ["kubernetes", "chart", "deployment", "template", "release"],
        "desc": "Kubernetes package manager"
    },
    "terraform": {
        "type": "tool", "domain": "devops",
        "connects": ["infrastructure_as_code", "aws", "gcp", "azure", "state", "provider"],
        "desc": "Infrastructure as Code tool"
    },
    "ci_cd": {
        "type": "concept", "domain": "devops",
        "connects": ["pipeline", "jenkins", "github_actions", "deployment", "testing", "automation"],
        "desc": "Continuous Integration and Deployment"
    },
    "prometheus": {
        "type": "tool", "domain": "devops",
        "connects": ["monitoring", "metrics", "alerting", "grafana", "time_series", "scraping"],
        "desc": "Monitoring and alerting system"
    },
    "grafana": {
        "type": "tool", "domain": "devops",
        "connects": ["prometheus", "dashboard", "visualization", "metrics", "alerting"],
        "desc": "Visualization and dashboarding tool"
    },

    # Python & Web
    "fastapi": {
        "type": "framework", "domain": "python",
        "connects": ["python", "api", "async", "pydantic", "openapi", "uvicorn", "rest"],
        "desc": "Modern Python web framework"
    },
    "asyncio": {
        "type": "library", "domain": "python",
        "connects": ["python", "async", "await", "coroutine", "event_loop", "concurrency"],
        "desc": "Python asynchronous I/O framework"
    },
    "pydantic": {
        "type": "library", "domain": "python",
        "connects": ["fastapi", "validation", "schema", "model", "python", "type_hints"],
        "desc": "Data validation using Python type hints"
    },
    "pytest": {
        "type": "tool", "domain": "python",
        "connects": ["testing", "python", "fixture", "assertion", "coverage", "mock"],
        "desc": "Python testing framework"
    },

    # Security
    "jwt": {
        "type": "standard", "domain": "security",
        "connects": ["authentication", "token", "oauth", "api", "claims", "signature"],
        "desc": "JSON Web Token for stateless authentication"
    },
    "oauth": {
        "type": "protocol", "domain": "security",
        "connects": ["authentication", "authorization", "jwt", "token", "scope", "client"],
        "desc": "Authorization framework"
    },
    "encryption": {
        "type": "concept", "domain": "security",
        "connects": ["aes", "rsa", "tls", "key", "cipher", "decrypt"],
        "desc": "Converting data to unreadable format"
    },
    "tls": {
        "type": "protocol", "domain": "security",
        "connects": ["https", "certificate", "encryption", "handshake", "ssl"],
        "desc": "Transport Layer Security for encrypted communication"
    },

    # API & Networking
    "api": {
        "type": "concept", "domain": "core",
        "connects": ["rest", "graphql", "endpoint", "request", "response", "fastapi", "authentication"],
        "desc": "Application Programming Interface"
    },
    "rest": {
        "type": "architecture", "domain": "api",
        "connects": ["api", "http", "json", "endpoint", "crud", "stateless"],
        "desc": "Representational State Transfer architectural style"
    },
    "graphql": {
        "type": "language", "domain": "api",
        "connects": ["api", "query", "mutation", "schema", "resolver", "apollo"],
        "desc": "Query language for APIs"
    },
    "http": {
        "type": "protocol", "domain": "networking",
        "connects": ["rest", "api", "request", "response", "status_code", "header", "tls"],
        "desc": "HyperText Transfer Protocol"
    },
    "websocket": {
        "type": "protocol", "domain": "networking",
        "connects": ["real_time", "bidirectional", "connection", "http", "socket"],
        "desc": "Full-duplex communication protocol"
    },

    # Cloud
    "aws": {
        "type": "platform", "domain": "cloud",
        "connects": ["ec2", "s3", "lambda", "rds", "terraform", "cloud"],
        "desc": "Amazon Web Services cloud platform"
    },
    "ec2": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "virtual_machine", "instance", "ami", "auto_scaling"],
        "desc": "AWS Elastic Compute Cloud"
    },
    "s3": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "storage", "bucket", "object", "cdn"],
        "desc": "AWS Simple Storage Service"
    },
    "lambda": {
        "type": "service", "domain": "cloud",
        "connects": ["aws", "serverless", "function", "event", "trigger"],
        "desc": "AWS serverless compute service"
    },

    # Universal super nodes
    "data": {
        "type": "concept", "domain": "core",
        "connects": ["database", "storage", "processing", "pipeline", "analytics", "ml", "json", "api"],
        "desc": "Information processed by systems"
    },
    "server": {
        "type": "resource", "domain": "core",
        "connects": ["api", "database", "docker", "kubernetes", "http", "request", "response", "load_balancer"],
        "desc": "Machine that processes requests"
    },
    "config": {
        "type": "concept", "domain": "core",
        "connects": ["environment", "yaml", "json", "settings", "terraform", "kubernetes", "docker"],
        "desc": "System configuration and settings"
    },
    "error": {
        "type": "concept", "domain": "core",
        "connects": ["exception", "logging", "debugging", "stack_trace", "handling", "monitoring", "alerting"],
        "desc": "System failures and exceptions"
    },
}

# Memory templates with semantic meaning
MEMORIES = [
    # ML
    ("fact", "neural_network", "Neural networks learn by adjusting weights through backpropagation"),
    ("fact", "transformer", "Transformers use self-attention to process sequences in parallel"),
    ("fact", "bert", "BERT is pre-trained on masked language modeling and next sentence prediction"),
    ("fact", "gradient_descent", "Gradient descent updates weights in the opposite direction of the gradient"),
    ("procedure", "overfitting", "To prevent overfitting, use dropout, regularization, and early stopping"),
    ("procedure", "neural_network", "Train a neural network by feeding batches and computing gradients"),

    # Database
    ("fact", "postgresql", "PostgreSQL supports ACID transactions and complex queries"),
    ("fact", "index", "B-tree indexes speed up equality and range queries"),
    ("fact", "redis", "Redis stores data in memory for sub-millisecond response times"),
    ("procedure", "sharding", "Implement sharding by partitioning data on a shard key"),
    ("procedure", "transaction", "Use transactions to ensure data consistency across multiple operations"),

    # DevOps
    ("fact", "docker", "Docker containers share the host kernel but have isolated filesystems"),
    ("fact", "kubernetes", "Kubernetes manages container deployment, scaling, and networking"),
    ("procedure", "ci_cd", "Set up CI/CD by defining build, test, and deploy stages in a pipeline"),
    ("procedure", "prometheus", "Configure Prometheus scraping to collect metrics from targets"),

    # Security
    ("fact", "jwt", "JWTs contain header, payload, and signature sections"),
    ("fact", "tls", "TLS encrypts data in transit using symmetric encryption after handshake"),
    ("procedure", "oauth", "Implement OAuth by registering client, getting authorization, exchanging tokens"),

    # API
    ("fact", "rest", "REST APIs use HTTP methods to perform CRUD operations on resources"),
    ("fact", "graphql", "GraphQL allows clients to request exactly the data they need"),
    ("procedure", "api", "Design APIs by defining resources, endpoints, and request/response schemas"),

    # Physics - Classical Mechanics
    ("fact", "newton_laws", "Newton's first law states an object remains at rest or in uniform motion unless acted upon by a force"),
    ("fact", "momentum", "Momentum is conserved in all collisions in an isolated system"),
    ("fact", "energy", "Energy cannot be created or destroyed, only transformed from one form to another"),
    ("fact", "gravity", "Gravitational force is proportional to the product of masses and inversely proportional to distance squared"),
    ("procedure", "lagrangian", "To derive equations of motion, compute the Lagrangian and apply Euler-Lagrange equations"),

    # Physics - Quantum
    ("fact", "quantum_mechanics", "Quantum mechanics describes the behavior of matter at atomic and subatomic scales"),
    ("fact", "wave_function", "The wave function's squared magnitude gives the probability density of finding a particle"),
    ("fact", "superposition", "Quantum systems can exist in multiple states simultaneously until measured"),
    ("fact", "entanglement", "Entangled particles remain correlated regardless of the distance between them"),
    ("fact", "heisenberg_uncertainty", "Position and momentum cannot both be precisely determined simultaneously"),
    ("procedure", "schrodinger_equation", "Solve the time-independent Schrödinger equation by finding eigenvalues of the Hamiltonian"),

    # Physics - Thermodynamics
    ("fact", "thermodynamics", "Thermodynamics governs energy transfer and transformation in physical systems"),
    ("fact", "entropy", "Entropy measures the number of microscopic configurations corresponding to a macroscopic state"),
    ("fact", "second_law", "The total entropy of an isolated system always increases over time"),
    ("fact", "temperature", "Temperature is a measure of the average kinetic energy of particles"),
    ("procedure", "statistical_mechanics", "Calculate macroscopic properties from the partition function using statistical mechanics"),

    # Physics - Electromagnetism
    ("fact", "electromagnetism", "Electric and magnetic fields are interrelated manifestations of the same force"),
    ("fact", "maxwell_equations", "Maxwell's equations unify electricity, magnetism, and optics"),
    ("fact", "electromagnetic_wave", "Electromagnetic waves propagate at the speed of light in vacuum"),
    ("fact", "photon", "A photon has energy proportional to its frequency: E = hν"),

    # Physics - Relativity
    ("fact", "special_relativity", "Time dilation and length contraction occur at speeds approaching light"),
    ("fact", "general_relativity", "Mass curves spacetime, and objects follow geodesics in curved spacetime"),
    ("fact", "mass_energy", "Mass and energy are equivalent, related by E = mc²"),
    ("fact", "black_hole", "A black hole forms when mass is compressed within its Schwarzschild radius"),

    # Physics - Particle
    ("fact", "standard_model", "The Standard Model describes 17 fundamental particles and their interactions"),
    ("fact", "quark", "Quarks combine in threes to form protons and neutrons via the strong force"),
    ("fact", "higgs", "The Higgs boson was discovered at CERN in 2012, confirming the Higgs mechanism"),

    # Math - Calculus
    ("fact", "calculus", "Calculus provides tools to analyze continuous change through derivatives and integrals"),
    ("fact", "derivative", "The derivative gives the instantaneous rate of change of a function"),
    ("fact", "integral", "The integral computes accumulated quantities like area under a curve"),
    ("fact", "limit", "Limits define the behavior of functions as inputs approach specific values"),
    ("procedure", "differential_equation", "Solve ODEs by separating variables or using integrating factors"),
    ("procedure", "taylor_series", "Approximate a function by computing derivatives at a point and summing the Taylor series"),

    # Math - Linear Algebra
    ("fact", "linear_algebra", "Linear algebra studies vector spaces and linear mappings between them"),
    ("fact", "eigenvalue", "Eigenvalues reveal special directions where linear transformations act by scaling"),
    ("fact", "matrix", "Matrix multiplication represents composition of linear transformations"),
    ("fact", "determinant", "A matrix is invertible if and only if its determinant is nonzero"),
    ("procedure", "eigenvector", "Find eigenvectors by solving (A - λI)v = 0 for each eigenvalue λ"),

    # Math - Probability
    ("fact", "probability", "Probability quantifies uncertainty using numbers between 0 and 1"),
    ("fact", "bayes_theorem", "Bayes' theorem updates beliefs based on new evidence: P(A|B) = P(B|A)P(A)/P(B)"),
    ("fact", "normal_distribution", "The normal distribution is fully characterized by mean and standard deviation"),
    ("fact", "variance", "Variance measures how spread out values are from the expected value"),
    ("procedure", "hypothesis_testing", "Conduct hypothesis tests by computing test statistics and comparing to critical values"),

    # Math - Abstract Algebra
    ("fact", "group", "A group is a set with an associative operation, identity element, and inverses"),
    ("fact", "ring", "A ring has addition (forming a group) and multiplication (associative with identity)"),
    ("fact", "field_algebra", "A field is a ring where every nonzero element has a multiplicative inverse"),
    ("fact", "homomorphism", "Homomorphisms preserve algebraic structure between mathematical objects"),

    # Math - Topology
    ("fact", "topology", "Topology studies properties preserved under continuous deformation"),
    ("fact", "manifold", "A manifold is a space that locally looks like Euclidean space"),
    ("fact", "homeomorphism", "Two spaces are topologically equivalent if there's a homeomorphism between them"),

    # Math - Number Theory
    ("fact", "prime", "The fundamental theorem of arithmetic states every integer factors uniquely into primes"),
    ("fact", "modular_arithmetic", "Modular arithmetic wraps numbers around after reaching the modulus"),

    # Math - Analysis
    ("fact", "real_analysis", "Real analysis provides rigorous foundations for calculus"),
    ("fact", "complex_analysis", "Analytic functions have derivatives that satisfy the Cauchy-Riemann equations"),

    # Math - Discrete
    ("fact", "graph_theory", "Graph theory studies networks of vertices connected by edges"),
    ("fact", "combinatorics", "Combinatorics counts and arranges discrete structures"),
    ("fact", "logic", "Mathematical logic formalizes valid reasoning and proof"),

    # Cross-domain connections
    ("relationship", "gradient_descent", "Gradient descent optimization connects calculus derivatives to machine learning"),
    ("relationship", "quantum_computing", "Quantum computing combines quantum mechanics with information theory"),
    ("relationship", "cryptography", "Modern cryptography relies on number theory, especially prime factorization"),
    ("relationship", "pca", "PCA uses eigenvalues to find principal components in data"),
    ("relationship", "neural_network", "Neural networks approximate functions using composition of simple operations"),
]


async def create_semantic_data():
    """Create semantically connected data."""
    db = Neo4jClient()
    await db.connect()

    print("Creating semantically connected knowledge graph...")

    concept_ids = {}

    # Create all concepts
    print("Creating concepts...")
    for name, info in KNOWLEDGE_GRAPH.items():
        concept_id = f"c_{name}_{uuid.uuid4().hex[:4]}"
        concept_ids[name] = concept_id

        # Position by domain
        domains = ["ml", "db", "devops", "python", "security", "api", "networking", "cloud", "core", "physics", "math"]
        domain_idx = domains.index(info["domain"]) if info["domain"] in domains else 0
        angle = (domain_idx / len(domains)) * 6.28 + random.uniform(-0.3, 0.3)
        radius = random.uniform(20000, 40000) if info["domain"] != "core" else random.uniform(5000, 15000)
        x = radius * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1)
        y = radius * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1)
        cluster = domain_idx

        await db.execute_query(
            """
            CREATE (c:Concept {
                id: $id, name: $name, type: $type, description: $desc,
                domain: $domain, activation_count: $act,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=concept_id, name=name, type=info["type"], desc=info["desc"],
            domain=info["domain"], act=random.randint(10, 200),
            x=x, y=y, cluster=cluster
        )

    print(f"Created {len(concept_ids)} concepts")

    # Create semantic relationships
    print("Creating semantic relationships...")
    rel_count = 0
    for name, info in KNOWLEDGE_GRAPH.items():
        for target in info["connects"]:
            if target in concept_ids:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=concept_ids[name], id2=concept_ids[target],
                    w=random.uniform(0.6, 1.0)
                )
                rel_count += 1

    print(f"Created {rel_count} semantic relationships")

    # Boost core concepts (super nodes) with more connections
    print("Boosting super nodes...")
    super_nodes = ["api", "data", "server", "config", "error", "energy", "function", "probability", "field", "conservation"]
    all_concept_ids = list(concept_ids.values())

    for super_name in super_nodes:
        super_id = concept_ids[super_name]
        # Connect to 80% of all concepts
        for cid in all_concept_ids:
            if cid != super_id and random.random() < 0.8:
                await db.execute_query(
                    """
                    MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                    MERGE (a)-[:RELATED_TO {weight: $w}]->(b)
                    """,
                    id1=cid, id2=super_id, w=random.uniform(0.3, 0.7)
                )

    # Add more concepts to reach 3000+ (3x original)
    print("Adding more concepts...")
    extra_concepts = []
    all_domains = ["ml", "db", "devops", "python", "security", "api", "networking", "cloud", "core", "physics", "math"]
    for base_name, info in list(KNOWLEDGE_GRAPH.items()):  # Use ALL base concepts
        for suffix in ["_v2", "_v3", "_advanced", "_config", "_utils", "_core", "_client", "_server",
                       "_handler", "_manager", "_service", "_module", "_plugin", "_extension",
                       "_theory", "_applied", "_intro", "_basics", "_fundamentals", "_principles",
                       "_impl", "_abstract", "_concrete", "_helper", "_wrapper", "_adapter",
                       "_factory", "_builder", "_observer", "_strategy", "_base", "_derived",
                       "_model", "_view", "_controller", "_interface", "_protocol", "_spec",
                       "_test", "_mock", "_stub", "_fixture", "_example", "_demo", "_tutorial",
                       "_simplified", "_optimized", "_parallel", "_distributed", "_cached",
                       "_async", "_sync", "_batch", "_stream", "_queue", "_event", "_hook"]:
            new_name = f"{base_name}{suffix}"
            concept_id = f"c_{new_name}_{uuid.uuid4().hex[:4]}"
            extra_concepts.append((new_name, concept_id, info["domain"]))

            domain_idx = all_domains.index(info["domain"]) if info["domain"] in all_domains else 0
            x = random.uniform(-50000, 50000)
            y = random.uniform(-50000, 50000)

            await db.execute_query(
                """
                CREATE (c:Concept {
                    id: $id, name: $name, type: 'variant', description: $desc,
                    domain: $domain, activation_count: $act,
                    layout_x: $x, layout_y: $y, cluster: $cluster
                })
                """,
                id=concept_id, name=new_name, desc=f"Extended {base_name} functionality",
                domain=info["domain"], act=random.randint(1, 50),
                x=x, y=y, cluster=domain_idx
            )

            # Connect to base concept
            await db.execute_query(
                """
                MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                CREATE (a)-[:RELATED_TO {weight: $w}]->(b)
                """,
                id1=concept_id, id2=concept_ids[base_name], w=random.uniform(0.7, 1.0)
            )

            # Connect to super nodes
            for super_name in super_nodes:
                if random.random() < 0.6:
                    await db.execute_query(
                        """
                        MATCH (a:Concept {id: $id1}), (b:Concept {id: $id2})
                        MERGE (a)-[:RELATED_TO {weight: $w}]->(b)
                        """,
                        id1=concept_id, id2=concept_ids[super_name], w=random.uniform(0.2, 0.5)
                    )

    print(f"Added {len(extra_concepts)} extra concepts")

    # Create semantic memories
    print("Creating memories...")
    memory_count = 0
    for mtype, concept, content in MEMORIES:
        if concept in concept_ids:
            memory_id = f"mem_{uuid.uuid4().hex[:8]}"
            x = random.uniform(-50000, 50000)
            y = random.uniform(-50000, 50000)

            await db.execute_query(
                """
                CREATE (m:SemanticMemory {
                    id: $id, content: $content, memory_type: $mtype,
                    importance: $importance, strength: $strength, access_count: $access,
                    layout_x: $x, layout_y: $y, cluster: $cluster
                })
                """,
                id=memory_id, content=content, mtype=mtype,
                importance=random.uniform(5, 10), strength=random.uniform(0.7, 1.0),
                access=random.randint(5, 50), x=x, y=y, cluster=random.randint(0, 10)
            )

            await db.execute_query(
                "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                mid=memory_id, cid=concept_ids[concept]
            )
            memory_count += 1

    # Add more memories (6000 for 3x data)
    for _ in range(6000):
        concept_name = random.choice(list(KNOWLEDGE_GRAPH.keys()))
        info = KNOWLEDGE_GRAPH[concept_name]

        templates = [
            f"{concept_name} is essential for {info['domain']} applications",
            f"When using {concept_name}, consider {random.choice(info['connects']) if info['connects'] else 'performance'}",
            f"{concept_name} integrates well with {random.choice(info['connects']) if info['connects'] else 'other tools'}",
            f"Best practice: configure {concept_name} for production use",
            f"Debug {concept_name} issues by checking logs and metrics",
        ]

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        x = random.uniform(-50000, 50000)
        y = random.uniform(-50000, 50000)

        await db.execute_query(
            """
            CREATE (m:SemanticMemory {
                id: $id, content: $content, memory_type: $mtype,
                importance: $importance, strength: $strength, access_count: $access,
                layout_x: $x, layout_y: $y, cluster: $cluster
            })
            """,
            id=memory_id, content=random.choice(templates),
            mtype=random.choice(["fact", "procedure", "relationship"]),
            importance=random.uniform(3, 9), strength=random.uniform(0.5, 1.0),
            access=random.randint(1, 30), x=x, y=y, cluster=random.randint(0, 10)
        )

        if concept_name in concept_ids:
            await db.execute_query(
                "MATCH (m:SemanticMemory {id: $mid}), (c:Concept {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                mid=memory_id, cid=concept_ids[concept_name]
            )
        memory_count += 1

    print(f"Created {memory_count} memories")

    # Check super node connections
    print("\nSuper node connections:")
    for name in super_nodes:
        result = await db.execute_query(
            "MATCH (c:Concept {id: $id})-[r]-() RETURN count(r) as cnt",
            id=concept_ids[name]
        )
        print(f"  {name}: {result[0]['cnt']} connections")

    await db.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(create_semantic_data())
