# Directory configuration
BUILD_DIR = build
IMAGES_DIR = imgs
PRESENTATION_DIR = presentation

# Directory path combining
PRESENTATION_BUILD_DIR = $(BUILD_DIR)/$(PRESENTATION_DIR)

# Images dir
IMAGES_SRCS := $(wildcard imgs/*.tex)

# Directory guard
dir_guard = @mkdir -p $(@D)

# All recipe
all: presentation

presentation: $(PRESENTATION_BUILD_DIR)/presentation.pdf

$(PRESENTATION_BUILD_DIR)/presentation.pdf: presentation.tex ${IMAGES_SRCS} ${SECTIONS_SRCS} ${CHAPTERS_SRCS}
	$(dir_guard)
	latexmk -shell-escape -output-directory=$(PRESENTATION_BUILD_DIR) $<

# Clean recipe
clean:
	rm -rf $(BUILD_DIR)
