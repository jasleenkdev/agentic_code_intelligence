# Thin wrappers around run.sh so both `make X` and `./run.sh X` work.
.PHONY: setup index app dev streamlit test e2e eval gold lint
setup: ; ./run.sh setup
index: ; ./run.sh index
app:   ; ./run.sh app
dev:   ; ./run.sh dev
streamlit: ; ./run.sh streamlit
test:  ; ./run.sh test
e2e:   ; ./run.sh e2e
eval:  ; ./run.sh eval
gold:  ; ./run.sh gold
lint:  ; ./run.sh lint
