> # About CI/CD pipeline
> - Merges to acceptance and main branch triggers GitHub actions in which an image is build and pushed to docker hub.
> - After succesfully building image, deploy job uses docker swarm to 'deploy stack deploy' which is a docker swarm utility.

## How to add models or change environment variables.
- Since models are large in size they are not included in docker image building process.And having .env files would be too insecure.
- Docker secret functionality explored but decided against since has too many unstable behaviour working with docker swarm.
- All things considered, the agreed upon way of updating/adding models and environment variables is by using software that support data transfer over SSH
- SSH is hardened to prevent any potential attack and can only accessed with encryption keys.(Password auth is disabled)
- After Development Team adding the requested ssh key to VPS's authorized keys, data can be transfered(e.g rsync, scp or you IDE's ssh extension)


> ### Disclaimers
> Development Team opted for 'Docker Swarm' because, it has the most potential for scaling up since it can handle most of the scaling operations kubernetes handles.
> Actually it wouldn't be too far fetched to compare docker swarm instances compare docker swarm instances With Kubernetes clustersWith Kubernetes clusters.
