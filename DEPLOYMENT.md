# ==============================================================================
# WeatherCast AI — Production Deployment Guide
# AWS ECR & ECS Fargate Production Infrastructure
# ==============================================================================

This document provides complete instructions for containerizing, testing, publishing, and deploying **WeatherCast AI** to **Amazon Web Services (AWS)** using **Amazon ECR** and **Amazon ECS Fargate** with an Application Load Balancer (ALB).

---

## 1. Project Overview & Architecture

WeatherCast AI is a 24-hour temperature forecasting system built with **Django 5.0+**, **XGBoost (C++ engine)**, and an interactive **Vanilla JavaScript/CSS** glassmorphism dashboard.

```
+-------------------------------------------------------------+
|                          Clients                            |
+-------------------------------------------------------------+
                              | HTTPS (443)
                              v
+-------------------------------------------------------------+
|               AWS Application Load Balancer                 |
+-------------------------------------------------------------+
                              | HTTP (8000)
                              v
+-------------------------------------------------------------+
|            Amazon ECS Fargate (Private Subnet)              |
|                                                             |
|  Container: weathercast-ai:v1.0.0                           |
|  - Non-root user: appuser (UID 1001)                        |
|  - Gunicorn WSGI Server (0.0.0.0:8000)                      |
|  - XGBoost ML Engine with OpenMP (libgomp1)                 |
|  - Outbound HTTPS to Open-Meteo API (NAT Gateway)           |
+-------------------------------------------------------------+
```

---

## 2. Local Development & Docker Testing

### 2.1 Prerequisites
- **Docker Desktop** installed on your workstation.
- **Python 3.12+** (if running bare-metal).

### 2.2 Local Container Build & Verification

```powershell
# [LOCAL WINDOWS TERMINAL]
# 1. Build the production Docker image
docker build -t weathercast-ai:v1.0.0 -f Dockerfile .

# 2. Run container locally with test environment variables
docker run -d `
  --name weathercast_local `
  -p 8000:8000 `
  -e DJANGO_SECRET_KEY="local-test-secret-key-32-characters-minimum" `
  -e DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1" `
  weathercast-ai:v1.0.0

# 3. Check container logs
docker logs -f weathercast_local

# 4. Check Healthcheck status
docker inspect --format='{{json .State.Health.Status}}' weathercast_local

# 5. Test health endpoint via curl or browser
curl http://127.0.0.1:8000/api/health/

# 6. Stop and remove local container
docker stop weathercast_local
docker rm weathercast_local
```

---

## 3. AWS ECR (Elastic Container Registry) Workflow

**Amazon ECR** is AWS's managed Docker container registry. Images built locally or in CI/CD pipelines are pushed to ECR so that ECS Fargate tasks can securely pull and execute them.

### 3.1 Required Placeholders

Replace the following placeholder values when executing commands:
- `<AWS_REGION>`: Your target AWS region (e.g., `eu-west-1` or `us-east-1`).
- `<AWS_ACCOUNT_ID>`: Your 12-digit AWS Account ID (e.g., `123456789012`).
- `ECR_REPOSITORY`: `weathercast-ai`
- `IMAGE_TAG`: `v1.0.0` (Semantic version)

### 3.2 Step-by-Step ECR Commands

```powershell
# [LOCAL WINDOWS TERMINAL] — Authenticate Docker with AWS ECR
aws ecr get-login-password --region <AWS_REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com

# [LOCAL WINDOWS TERMINAL] — Create ECR Repository (if it doesn't already exist)
aws ecr create-repository `
  --repository-name weathercast-ai `
  --region <AWS_REGION> `
  --image-scanning-configuration scanOnPush=true `
  --encryption-configuration encryptionType=AES256

# [LOCAL WINDOWS TERMINAL] — Build the Docker image locally
docker build -t weathercast-ai:v1.0.0 -f Dockerfile .

# [LOCAL WINDOWS TERMINAL] — Tag the image for ECR
docker tag weathercast-ai:v1.0.0 <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/weathercast-ai:v1.0.0
docker tag weathercast-ai:v1.0.0 <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/weathercast-ai:latest

# [LOCAL WINDOWS TERMINAL] — Push images to ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/weathercast-ai:v1.0.0
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/weathercast-ai:latest

# [LOCAL WINDOWS TERMINAL] — Verify that image exists in ECR
aws ecr describe-images --repository-name weathercast-ai --region <AWS_REGION>
```

---

## 4. AWS IAM & Permissions Architecture (Least Privilege)

### 4.1 Deployment Machine / CI IAM Policy
The IAM user or CI/CD role running the build and push commands requires only minimal ECR permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRPushOperations",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeImages"
      ],
      "Resource": "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/weathercast-ai"
    }
  ]
}
```

### 4.2 ECS Task Execution Role (`ecsTaskExecutionRole`)
Allows the ECS infrastructure agent to pull the image from ECR and stream container stdout/stderr logs to Amazon CloudWatch:
- AWS Managed Policy: `arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy`
- If pulling secrets from AWS Secrets Manager: Add `secretsmanager:GetSecretValue` permission for the specific secret ARN.

### 4.3 ECS Task Role (`weathercastTaskRole`)
Allows the running Django application code itself to interact with AWS resources. Because WeatherCast AI does not call internal AWS APIs at runtime, this role requires **no permissions attached**, maintaining strict zero-trust posture.

---

## 5. Deployment Options Comparison

| Option | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **AWS ECS Fargate** | Fully serverless containers, zero OS maintenance, autoscaling, integrates natively with ALB & Secrets Manager. | Slightly slower cold starts compared to pre-warmed VMs. | **Recommended for Production** |
| **AWS App Runner** | Simplest setup, automatic SSL/TLS & scaling. | Limited VPC configuration flexibility for enterprise architectures. | Alternative for single-service prototypes |
| **EC2 + Docker** | Maximum OS access and control. | Manual patching, high operational overhead, scaling complexity. | Not recommended |

**Why ECS Fargate?** WeatherCast AI is a stateless ML microservice with high CPU burst during XGBoost inference. ECS Fargate provisions exact CPU/Memory slices on-demand with zero server management.

---

## 6. Production Deployment on AWS ECS Fargate

### 6.1 Architecture Components

1. **VPC**: 2 Public Subnets (for ALB) + 2 Private Subnets (for Fargate Tasks with NAT Gateway).
2. **Security Groups**:
   - `alb-sg`: Inbound TCP 80/443 from `0.0.0.0/0`. Outbound TCP 8000 to `fargate-sg`.
   - `fargate-sg`: Inbound TCP 8000 from `alb-sg` ONLY. Outbound TCP 443 to `0.0.0.0/0` (for Open-Meteo external API calls).
3. **Application Load Balancer (ALB)**:
   - Target Group: Port 8000, Protocol HTTP, Target Type `IP`.
   - Health check path: `/api/health/`, Port: `traffic-port`, Interval: 30s, Timeout: 5s, Healthy Threshold: 2, Success codes: `200`.
4. **Task Definition Specifications**:
   - Launch type: `FARGATE`
   - CPU: `0.5 vCPU` (512 units)
   - Memory: `1 GB` (1024 MB)
   - Image: `<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/weathercast-ai:v1.0.0`
   - Port Mapping: Container Port `8000` (TCP)
   - Environment Variables:
     - `DJANGO_SETTINGS_MODULE`: `config.settings.production`
     - `DJANGO_ALLOWED_HOSTS`: `<YOUR_DOMAIN>,<ALB_DNS_NAME>`
   - Secrets (from AWS Secrets Manager):
     - `DJANGO_SECRET_KEY`: Value from `arn:aws:secretsmanager:<REGION>:<ACCOUNT>:secret:weathercast/django-secret-key`

---

## 7. Environment Variables Reference

| Variable | Required | Default / Example | Purpose |
| :--- | :--- | :--- | :--- |
| `DJANGO_SETTINGS_MODULE` | Yes | `config.settings.production` | Selects Django production security settings |
| `DJANGO_SECRET_KEY` | Yes | *(Random 64-char string)* | Django cryptographic signing key |
| `DJANGO_ALLOWED_HOSTS` | Yes | `api.example.com,weathercast-alb.amazonaws.com` | Prevents HTTP Host header attacks |
| `DJANGO_DEBUG` | Optional | `False` | Must remain `False` in production |
| `PORT` | Optional | `8000` | Application listening port |

---

## 8. Versioning & Release Workflow

### 8.1 Why Immutable Tags Matter
Never deploy raw `latest` to production. If an autoscaling event triggers or a container crashes, ECS pulls the image tag. Using `latest` creates ambiguity and prevents instant, reliable rollbacks.

### 8.2 Safe Release Workflow
```
Code Change
  --> Run Tests locally (pytest, smoke_test)
  --> Build Docker image with semantic tag (e.g. v1.1.0) and Git SHA
  --> Push v1.1.0 to Amazon ECR
  --> Create new Task Definition revision pointing to v1.1.0
  --> Update ECS Service (Rolling update without downtime)
  --> ECS verifies Healthcheck (/api/health/) on new tasks before draining old ones
  --> If unhealthy, rollback to v1.0.0 instantly with zero downtime
```
