docker cp ./network %1:/home/jovyan/work
docker cp ./model %1:/home/jovyan/work
docker cp ./checkpoint %1:/home/jovyan/work

docker cp ./imageTrain %1:/home/jovyan/work
docker cp ./imageTrain2 %1:/home/jovyan/work
docker cp ./imageValidate %1:/home/jovyan/work
docker cp ./imageValidate2 %1:/home/jovyan/work

docker cp ./utility %1:/home/jovyan/work