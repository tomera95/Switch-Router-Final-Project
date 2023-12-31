# Switch-Router-Final-Project
This is a final project of Switch and Router Architectures
course provided by the Hebrew University of Jerusalem.

In this repo we provide a simulation of our algorithms implementations
constructed by AIFO and SP-PIFO. The first algorithm is called 
"AIFO with FIFO" which consists AIFO with additional FIFO queue. The second 
algorithm is SP-AIFO, which operates as SP-PIFO with quantile as the priority of a queue.

#### First Simulation
The first simulation compares all 4 algorithms: AIFO, AIFO with FIFO ,SP-AIFO
and PIFO. In this simulation we sum the rank of packets advanced as a function of the number of packets. This simulation's purpose is to determine which algorithm has best results regarding prioritizing low ranked (high priority) packets.
###### Results
![](results/priority_compare.png)

#### Second Simulation
The second simulation compares "dropping" algorithms: AIFO, AIFO with FIFO and SP-AIFO.
In this simulation we check how many packets were dropped as a function of the capacity of AIFO and AIFO with FIFO. Using this simulation we would determine if increasing capacity changes the amount of packets dropped.
###### Results
![](results/thrown_compare.png)

#### Third Simulation
The final simulation is similar to the second. The difference is that we change the time of arrival of packets instead of the capacity
###### Results
![](results/thrown_compare_time.png)