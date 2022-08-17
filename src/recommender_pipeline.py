from kubernetes import client as k8s_client

client = kfp.Client()
exp = client.get_experiment(experiment_name ='mdupdate')

@dsl.pipeline(
    name='Recommender model update',
    description='Demonstrate usage of pipelines for multi-step model update'
)
def recommender_pipeline():
    # Load new data
    data = dsl.ContainerOp(
        name='updatedata',
        image='lightbend/recommender-data-update-publisher:0.2') \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_URL',
                                              value='http://minio-service.kubeflow.svc.cluster.local:9000')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_KEY', value='minio')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_SECRET', value='minio123'))
    # Train the model
    train = dsl.ContainerOp(
        name='trainmodel',
        image='lightbend/ml-tf-recommender:0.2') \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_URL',
                                              value='minio-service.kubeflow.svc.cluster.local:9000')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_KEY', value='minio')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_SECRET', value='minio123'))
    train.after(data)
    # Publish new model
    publish = dsl.ContainerOp(
        name='publishmodel',
        image='lightbend/recommender-model-publisher:0.2') \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_URL',
                                              value='http://minio-service.kubeflow.svc.cluster.local:9000')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_KEY', value='minio')) \
        .add_env_variable(k8s_client.V1EnvVar(name='MINIO_SECRET', value='minio123')) \
        .add_env_variable(k8s_client.V1EnvVar(name='KAFKA_BROKERS',
                                              value='cloudflow-kafka-brokers.cloudflow.svc.cluster.local:9092')) \
        .add_env_variable(k8s_client.V1EnvVar(name='DEFAULT_RECOMMENDER_URL',
                                              value='http://recommendermodelserver.kubeflow.svc.cluster.local:8501')) \
        .add_env_variable(k8s_client.V1EnvVar(name='ALTERNATIVE_RECOMMENDER_URL',
                                              value='http://recommendermodelserver1.kubeflow.svc.cluster.local:8501'))
    publish.after(train)

