# make sync jupytext <path>
# make sync marimo   <path>
# make open jupyter <path>
# make open marimo  <path>

MODE := $(word 2,$(MAKECMDGOALS))
FILE := $(word 3,$(MAKECMDGOALS))

.PHONY: sync
sync:
	@mode="$(MODE)"; file="$(FILE)"; \
	if [ -z "$$mode" ] || [ -z "$$file" ]; then \
		echo "Usage: make sync jupytext|marimo <path-to-file>"; \
		exit 2; \
	fi; \
	if [ ! -e "$$file" ]; then echo "No such file: $$file"; exit 2; fi; \
	case "$$mode" in \
	  jupytext) \
		svc=$$(docker compose ps --status running --services | head -1); \
		if [ -z "$$svc" ]; then \
			echo "No kaggle container running. Start one with:"; \
			echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
			exit 1; \
		fi; \
		dir=$$(dirname "$$file"); \
		case "$$file" in \
			*.ipynb) docker compose exec --user "$$(id -u):$$(id -g)" "$$svc" \
				jupytext --quiet --use-source-timestamp --to py:percent \
				--output "$${file%.ipynb}.py" "$$file" ;; \
			*) docker compose exec --user "$$(id -u):$$(id -g)" "$$svc" jupytext --sync "$$file" ;; \
		esac; \
		uv run scripts/generate_metadata.py --file "$$dir" --competition "$$comp" --owner "$$owner" --slug "$$slug" ;; \
	  marimo) \
		dir=$$(dirname "$$file"); \
		base=$$(basename "$$file"); \
		nb_file="$$dir/$${base%.*}_nb.py"; \
		docker exec kaggle-notebooks-gpu marimo convert "$$file" -o "$$nb_file"; \
		echo "Generated $$nb_file" ;; \
	  *) echo "Unknown sync mode: $$mode (expected jupytext or marimo)"; exit 2 ;; \
	esac

.PHONY: open
open:
	@mode="$(MODE)"; file="$(FILE)"; \
	if [ -z "$$mode" ] || [ -z "$$file" ]; then \
		echo "Usage: make open jupyter|marimo <path-to-file>"; \
		exit 2; \
	fi; \
	case "$$mode" in \
	  jupyter) \
		svc=$$(docker compose ps --status running --services | head -1); \
		if [ -z "$$svc" ]; then \
			echo "No kaggle container running. Start one with:"; \
			echo "  docker compose --profile cpu up -d   # or --profile gpu"; \
			exit 1; \
		fi; \
		dir=$$(dirname "$$file"); \
		slug=$$(basename "$$dir"); \
		notebook="$$dir/$$slug.ipynb"; \
		if [ ! -f "$$notebook" ]; then \
			if [ -f "$$dir/$$slug.py" ]; then \
				$(MAKE) --no-print-directory sync jupytext "$$dir/$$slug.py" || exit $$?; \
			else \
				echo "No notebook found at $$notebook"; exit 2; \
			fi; \
		fi; \
		url="http://localhost:$${JUPYTER_PORT:-8888}/lab/tree/$$notebook"; \
		echo "Opening $$url"; \
		xdg-open "$$url" >/dev/null 2>&1 & \
		;; \
	  marimo) \
		dir=$$(dirname "$$file"); \
		base=$$(basename "$$file"); \
		case "$$base" in \
			*_nb.py) nb_file="$$file" ;; \
			*) nb_file="$$dir/$${base%.*}_nb.py" ;; \
		esac; \
		if [ ! -f "$$nb_file" ]; then \
			$(MAKE) --no-print-directory sync marimo "$$file" || exit $$?; \
		fi; \
		docker exec -it kaggle-notebooks-gpu marimo edit --host 0.0.0.0 --port 2718 "$$nb_file" \
		;; \
	  *) echo "Unknown open mode: $$mode (expected jupyter or marimo)"; exit 2 ;; \
	esac

# swallow extra positional args (mode + path) as no-op targets
%:
	@:
