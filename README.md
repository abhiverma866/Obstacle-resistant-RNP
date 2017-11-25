# Obstacle-resistant-RNP
Obstacle resistant Relay Node Placement using Computational Geometry technique (Alpha Shapes and Triangulation)

The objective of this work is to develop a strategy for relay node placement in
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


Research papers links:
Relay node placement techniques in wireless sensor networks
ieeexplore.ieee.org/iel7/7370053/7380415/07380683.pdf

AN OBSTACLE-RESISTANT RELAY NODE PLACEMENT IN CONSTRAINED ENVIRONMENT
ieeexplore.ieee.org/document/7938976/

Relay Node Placement in Constrained Environment
In book: Communication and Computing Systems, Chapter: Communication and Computing Systems, Publisher: Taylor & Francis Group, 6000 Broken Sound Parkway NW, Suite 300, Boca Raton, FL 33487-2742 CRC Press 2016, Editors: B.M.K. Prasad Krishna Kant Singh Neelam Ruhil Karan Singh Richard O’Kennedy, pp.417-420 

https://books.google.co.in/books?id=EPYgDgAAQBAJ&pg=PA417&lpg=PA417&dq=relay+node+placement+in+constrained+environment+taylor+and+francis&source=bl&ots=J-aesld4MP&sig=R-Wyje599SgnjnEH_2RXn1pLDg0&hl=en&sa=X&ved=0ahUKEwiIqbrHitrXAhUJuI8KHVX_BsUQ6AEIMTAB#v=onepage&q=relay%20node%20placement%20in%20constrained%20environment%20taylor%20and%20francis&f=false
