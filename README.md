# Scheduling Parameters with [ReLax](https://github.com/nslyubaykin/relax) (TRPO step KL divergence)

This repository contains a demonstration of scheduling possibilities in [ReLax](https://github.com/nslyubaykin/relax) (TRPO step KL divergence). 
Plot below shows a theoretical (scheduled) step KL-divergence versus an actual (derived with estimating Fisher vector product) for TRPO-GAE algorithm. 
This schedule is sub-optimal in terms of training performance and built for demonstration purposes only.

![kl_div_plot](https://github.com/nslyubaykin/trpo_schedule_kl/blob/master/kl_div_plot.png)
