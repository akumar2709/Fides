The above code can be used to generate a plot of divergence using Wasserstien Metric and Jeffery's Divergence. To generate the plot run - 
```bash
	$   python divergence_plot.py [divergence measure] [model and dataset]
```

Possible arguments for [divergence measurement] = kld, w
possible arguments for [model and dataset] = resnet10, densenet10, efficientnet10, resnet100, densenet100, efficientnet100

Example run - 
```bash
	$   python divergence_plot.py kld resnet10
```
