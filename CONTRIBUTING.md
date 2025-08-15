# Contributing to Kenyan Fraud Detection System

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template when creating a new issue
- Include detailed steps to reproduce the bug
- Include any relevant logs or screenshots

### Suggesting Enhancements

- Check if the enhancement has already been suggested in the Issues section
- Use the feature request template when creating a new issue
- Explain why this enhancement would be useful to most users

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Install the package in development mode with `pip install -e .`

## Testing

Run tests with:

```bash
pytest tests/
```

## Code Style

This project follows the PEP 8 style guide for Python code. You can use tools like `flake8` and `black` to ensure your code adheres to these standards:

```bash
# Check code style
flake8 src/

# Format code
black src/ tests/
```

## Documentation

- Update documentation to reflect your changes
- Use docstrings for all functions, classes, and methods
- Follow the Google Style Python Docstrings format

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests liberally after the first line

## Additional Notes

### Git Workflow

We use the [GitHub flow](https://guides.github.com/introduction/flow/) for collaboration:
1. Create a branch from `main`
2. Add commits with your changes
3. Open a pull request
4. Discuss and review your code
5. Merge to `main` once approved

Thank you for contributing!
