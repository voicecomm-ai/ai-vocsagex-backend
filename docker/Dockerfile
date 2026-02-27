FROM ai-voicesagex-backend-base

WORKDIR /app

# COPY
COPY ./ /app/
COPY ./docker/docker-entrypoint.sh /app/
COPY ./docker/docker-env.sh /app/

# ENTRYPOINT
ENTRYPOINT ["./docker-entrypoint.sh"]