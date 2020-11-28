# Build an HTML version of the README.
README.html: README.md
	pandoc -s -o README.html -f commonmark README.md
