@Library("security_stages") _

pipeline {
    agent any
    options {
        buildDiscarder(logRotator(numToKeepStr: "3", artifactNumToKeepStr: "3"))
    }
    stages {
        stage('Setup') { // Install any dependencies you need to perform testing
            steps {
                script {
                sh """
                poetry install --extras "ui vector-stores-qdrant" --no-root
                """
                }
            }
        }
        stage ("Attempting security stages") {
            steps {
                shared()
            }
        }
        stage('Building our image') {
            steps {
                sh "docker build -f Dockerfile.local -t vinnimous/privategpt:latest ."
            }
        }
        stage('Login to Docker') {
            when {
                allOf {
                    branch 'master'
                }
            }
            steps {
                withCredentials([string (credentialsId: 'docker_hub_token', variable: 'token')]) {
                    sh "docker login --username vinnimous --password ${token}"
                }
            }
        }
        stage('Push to Docker') {
            when {
                allOf {
                    branch 'master'
                }
            }
            steps {
                sh "docker image push vinnimous/privategpt:latest"
            }
        }                                
    }
}
