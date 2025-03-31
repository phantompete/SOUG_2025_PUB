# SOUG_2025_PUB
This repository holds the demo for SOUG_2025.

Training a model for fuel consumption prediction based on car features. 
Saving prediction results into OCI Database with PostgreSQL 

## Components
- OCI Streaming
- OCI Data Flow
- OCI Database with PostgreSQL
- OCI Object Storage
- OCI Compute

## Architecture 
![Solution_Overview](/images/Solution_Overview.png)


## Functionality
### Option 1 - Batch 
- Run trainining batch application
- Run predict batch application
- Observe saved results in DB

### Option 2
- Run trainining batch application
- Run data streaming application (Optional - Fake kafka stream) 
- Run inference streaming application
- Observe saved results in DB

