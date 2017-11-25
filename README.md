# Obstacle-resistant-RNP
Obstacle resistant Relay Node Placement using Computational Geometry technique (Alpha Shapes and Triangulation)

The objective of this dissertation work is to develop a strategy for relay node placement in
constrained environment (i.e. obstacles). Two solutions have been proposed, first one is for
estimation of obstacles shape and second for relay node placement with obstacle avoidance.
Obstacle estimation is based on the Delaunay Triangulation which approximates the obstacles
shape present in the environment. The proposed solution for obstacle estimation does not require
any additional hardware (mobile robots) to estimate node locations thus can significantly reduce
network deployment costs. Various obstacles shapes have been estimated to show the accuracy of
the proposed solution. Obstacle estimation procedure estimates unpredictable obstacles with
higher accuracy and in very less time complexity.
The proposed solution for relay node placement in constrained environment is based on the
obstacle avoidance mechanism. In the first step the proposed solution estimates the locations which
are to be avoided for relay node deployment using the output of obstacle estimation procedure (i.e.
boundary point of obstacles). In second step Minimum Steiner Tree is constructed on surviving
(alive) sensor nodes avoiding the obstacle covered region. The proposed solution computes the
steiner points which are the estimated locations of relay nodes. The solution has been implemented
and tested for accuracy. However, there are many possible ways to improve the efficiency of both
the proposed solutions in terms of space and time complexity.
