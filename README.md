Commonly referred to as the third generation of neural networks, Spiking Neural
 Networks (SNNs) have attracted plenty of research interest in the last decade mainly
 due to its energy efficient and biologically realistic approach. Although areas like
 computer vision and signal processing have benefited significantly from SNNs, it
 seems NLP is still uncharted territory in neuromorphic devices. Our research seeks
 to establish the capability of Spike-Timing-Dependent Plasticity (R-STDP) in SNNs
 to conduct sentiment analysis. R-STDP provides a reward based learning mecha
nism that adjusts the synaptic weights according to the spike timing and feedback
 such as classification accuracy. This duplicates dopamine-controlled learning in the
 human brain. We also employed the Forward-Forward algorithm which replaces
 traditional backpropagation with local, layer-wise learning based on positive and
 negative sample contrast, allowing for modular and decentralized training without
 the need for backward error signals which further enhances biological plausibility. In
 addition, we employ an optimized rate coding method to convert textual data into
 spike trains that can then be easily processed by SNN architectures. We show that
 by applying this model on a benchmark sentiment analysis and affective computing
 dataset, SNNs, using learning rules such as R-STDP, can harness energy efficiency
 and the event-based nature of neuromorphic platforms to achieve sentiment classifi
cation accuracy (48%, 73%) comparable to conventional approaches. To understand
 the findings of our work, we compare our model to the existing deep learning models.
 The results obtained are of particular interest in order to assess the performance of
 spiking models for low-power NLP tasks, and to tailor SNNs into further machine
 learning pipelines.
