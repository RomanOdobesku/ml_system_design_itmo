# pylint: disable=expression-not-assigned

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.storage import S3
from diagrams.generic.compute import Rack
from diagrams.onprem.client import Users
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.mlops import Mlflow
from diagrams.programming.framework import Fastapi

# Define the architecture diagram
with Diagram(
    "ML Service Architecture",
    show=False,
    direction="LR",
    filename="ml_service_architecture_diagram",
):
    # External User sending requests
    user = Users("Client/API")

    # Gunicorn as load balancer
    gunicorn = Rack("Gunicorn (Distributes Load)")

    # Group of FastAPI workers
    with Cluster("Uvicorn Workers"):
        fastapi_workers = [Fastapi(f"Worker {i + 1}") for i in range(4)]

    # MLFlow and related components
    mlflow = Mlflow("MLFlow Model Registry")
    s3 = S3("Minio (Artifacts)")
    postgres = PostgreSQL("PostgreSQL (Metadata)")

    # Connections between components
    user << Edge(label='Request "/predict"') << gunicorn
    gunicorn >> Edge(label="") >> fastapi_workers
    fastapi_workers >> Edge(label="Fetch Model") >> mlflow
    mlflow >> Edge(label="Artifacts Storage") >> s3
    mlflow >> Edge(label="Metadata") >> postgres
    fastapi_workers >> Edge(label="") >> gunicorn
    gunicorn << Edge(label="Response") << user
