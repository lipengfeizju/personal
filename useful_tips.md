## Change the configuration to UCR webpage

In the `_config.yml` file, make the following changes

```yml
url: https://cs.ucr.edu/ # the base hostname & protocol for your site
baseurl: /~pli081 # the subpath of your site, e.g. /blog/. Leave blank for root
```

## Recompile the files

Launch the docker compose

```bash
docker compose up
```

Or the slim version for quick luaching

```bash
docker compose -f docker-compose-slim.yml up
```

## Transfer the deployable files to the UCR server

rsync --archive --partial --progress ./\_site homepage:/home/csgrads/pli081/

## Check the syntax

npx prettier --write \*.md
