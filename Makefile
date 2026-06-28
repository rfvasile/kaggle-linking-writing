# Optional positional arguments:
#   make sync <notebook>
#   make sync <competition> <owner/notebook-slug>
#   make download <competition>
#   make download <competition> <owner/notebook-slug>
#   make push <competition> <owner/notebook-slug>
#   make submit <competition> [path/to/submission.csv] MSG="message"
#   make open <competition> <owner/notebook-slug>
# Add FORCE=1 to overwrite existing download output.
ARG := $(word 2,$(MAKECMDGOALS))
ARG2 := $(word 3,$(MAKECMDGOALS))
EXTRA_GOALS := $(filter-out sync download push submit open,$(MAKECMDGOALS))
MSG ?=
DRY ?=

.PHONY: push
push:
	@if [ "$$(id -u)" -eq 0 ]; then \
		echo "Do not run make push with sudo; configure Docker access for your user so files and .cache stay user-owned."; \
		exit 2; \
	fi; \
	svc=$$(docker compose ps --status running --services | head -1); \
	if [ -z "$$svc" ]; then \
		echo "No kaggle container running. Start one with:"; \
		echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
		exit 1; \
	fi; \
	comp="$(ARG)"; \
	kernel="$(ARG2)"; \
	if [ -z "$$comp" ] || [ -z "$$kernel" ]; then \
		echo "Usage: make push <competition-slug> <owner/notebook-slug>"; exit 2; \
	fi; \
	notebook="notebooks/competitions/$$comp/$$kernel/$${kernel##*/}.py"; \
	if [ -f "$$notebook" ]; then \
		make --no-print-directory sync "$$notebook" || exit $$?; \
	fi; \
	dry_arg=""; \
	case "$(DRY)" in 1|true|TRUE|yes|YES) dry_arg="--dry-run" ;; esac; \
	if [ -z "$$dry_arg" ]; then \
		if [ -f .envrc.local ]; then set -a; . ./.envrc.local; set +a; fi; \
		if [ -z "$${KAGGLE_API_TOKEN:-}" ]; then \
			echo "No Kaggle credentials found. Add KAGGLE_API_TOKEN to .envrc.local."; \
			exit 2; \
		fi; \
	fi; \
	image_id=$$(docker container inspect "kaggle-notebooks-$$svc" --format '{{.Image}}'); \
	docker_image=$$(docker image inspect "$$image_id" --format '{{index .RepoDigests 0}}'); \
	if [ -z "$$docker_image" ]; then \
		echo "Could not resolve the repo digest of the running $$svc image."; exit 1; \
	fi; \
	docker compose exec --user "$$(id -u):$$(id -g)" \
		-e KAGGLEHUB_CACHE=/kaggle/working/.cache/kagglehub \
		-e KAGGLE_API_TOKEN \
		-e KAGGLE_DOCKER_IMAGE="$$docker_image" \
		"$$svc" python scripts/kaggle_push.py $$dry_arg "$$comp" "$$kernel"

.PHONY: submit
submit:
	@if [ "$$(id -u)" -eq 0 ]; then \
		echo "Do not run make submit with sudo; configure Docker access for your user so files and .cache stay user-owned."; \
		exit 2; \
	fi; \
	svc=$$(docker compose ps --status running --services | head -1); \
	if [ -z "$$svc" ]; then \
		echo "No kaggle container running. Start one with:"; \
		echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
		exit 1; \
	fi; \
	comp="$(ARG)"; \
	file="$(ARG2)"; \
	if [ -z "$$comp" ]; then \
		echo "Usage: make submit <competition-slug> [path/to/submission.csv] MSG=\"message\""; exit 2; \
	fi; \
	if [ -f .envrc.local ]; then set -a; . ./.envrc.local; set +a; fi; \
	if [ -z "$${KAGGLE_API_TOKEN:-}" ]; then \
		echo "No Kaggle credentials found. Add KAGGLE_API_TOKEN to .envrc.local."; \
		exit 2; \
	fi; \
	docker compose exec --user "$$(id -u):$$(id -g)" \
		-e KAGGLEHUB_CACHE=/kaggle/working/.cache/kagglehub \
		-e KAGGLE_API_TOKEN \
		"$$svc" python scripts/kaggle_submit.py "$$comp" $$file --message "$(MSG)"

.PHONY: open
open:
	@svc=$$(docker compose ps --status running --services | head -1); \
	if [ -z "$$svc" ]; then \
		echo "No kaggle container running. Start one with:"; \
		echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
		exit 1; \
	fi; \
	comp="$(ARG)"; \
	kernel="$(ARG2)"; \
	if [ -z "$$comp" ] || [ -z "$$kernel" ]; then \
		echo "Usage: make open <competition-slug> <owner/notebook-slug>"; exit 2; \
	fi; \
	notebook="notebooks/competitions/$$comp/$$kernel/$${kernel##*/}.ipynb"; \
	if [ ! -f "$$notebook" ]; then \
		if [ -f "$${notebook%.ipynb}.py" ]; then \
			make --no-print-directory sync "$$comp" "$$kernel" || exit $$?; \
		else \
			echo "No notebook found at $$notebook"; exit 2; \
		fi; \
	fi; \
	url="http://localhost:$${JUPYTER_PORT:-8888}/lab/tree/$$notebook"; \
	echo "Opening $$url"; \
	xdg-open "$$url" >/dev/null 2>&1 &

ifneq ($(filter sync download push submit open,$(MAKECMDGOALS)),)
ifneq ($(EXTRA_GOALS),)
.PHONY: $(EXTRA_GOALS)
$(EXTRA_GOALS):
	@:
endif
endif
