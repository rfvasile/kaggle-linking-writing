# Optional positional arguments:
#   make sync <competition> <owner/notebook-slug>
#   make open <competition> <owner/notebook-slug>
ARG := $(word 2,$(MAKECMDGOALS))
ARG2 := $(word 3,$(MAKECMDGOALS))
EXTRA_GOALS := $(filter-out sync open,$(MAKECMDGOALS))

.PHONY: sync
sync:
	@svc=$$(docker compose ps --status running --services | head -1); \
	if [ -z "$$svc" ]; then \
		echo "No kaggle container running. Start one with:"; \
		echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
		exit 1; \
	fi; \
	name="$(ARG)"; \
	kernel="$(ARG2)"; \
	if [ -z "$$name" ] || [ -z "$$kernel" ]; then \
		echo "Usage: make sync ARG=<competition> ARG2=<owner>/<notebook>"; \
		exit 2; \
	fi; \
	slug="$${kernel##*/}"; \
	base="notebooks/competitions/$$name/$$kernel/$$slug"; \
	if [ -e "$$base.py" ]; then file="$$base.py"; \
	elif [ -e "$$base.ipynb" ]; then file="$$base.ipynb"; \
	else echo "No paired notebook found at $$base (.py or .ipynb)"; exit 2; fi; \
	case "$$file" in \
		*.ipynb) docker compose exec --user "$$(id -u):$$(id -g)" "$$svc" \
			jupytext --quiet --use-source-timestamp --to py:percent \
			--output "$${file%.ipynb}.py" "$$file" ;; \
		*) docker compose exec --user "$$(id -u):$$(id -g)" "$$svc" jupytext --sync $$file ;; \
	esac; \
	uv run scripts/generate_metadata.py --competition "$$name" --kernel "$$kernel"


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
