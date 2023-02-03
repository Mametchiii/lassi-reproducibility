
LASSI is a state-of-the-art representation learning method for enforcing and certifying the individual fairness of
high-dimensional data, such as images. LASSI defines the set of similar individuals in the latent space of
generative Glow models, which allows us to capture and modify complex continuous semantic attributes
(e.g., hair color).  Next, LASSI employs adversarial learning and center smoothing to learn representations that
provably map similar individuals close to each other. Finally, it uses randomized smoothing to verify local
robustness of downstream applications, resulting in an individual fairness certificate of the end-to-end model.