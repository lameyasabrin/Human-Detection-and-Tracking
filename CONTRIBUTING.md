# Contributing to Human Detection and Tracking

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/human-detection-tracking.git
cd human-detection-tracking

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/human-detection-tracking.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- Clear, descriptive title
- Detailed description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Screenshots or error logs if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- Clear description of the enhancement
- Use cases and benefits
- Possible implementation approach
- Any relevant examples or references

### Code Contributions

Areas where contributions are especially welcome:

- New detection models
- Improved tracking algorithms
- Performance optimizations
- Documentation improvements
- Bug fixes
- Test coverage
- Example notebooks

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use descriptive variable names
good: detection_confidence = 0.8
bad:  dc = 0.8

# Add type hints
def detect_humans(frame: np.ndarray, threshold: float = 0.5) -> List[Detection]:
    pass

# Document functions with docstrings
def process_video(video_path: str) -> Dict:
    """
    Process video file and return analytics.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Dictionary containing processing results
    """
    pass
```

### Code Formatting

We use automated formatters:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run before committing:

```bash
black src/
isort src/
flake8 src/
```

### Documentation

- Add docstrings to all public functions and classes
- Update README.md if adding new features
- Add inline comments for complex logic
- Create examples in `demo/` for new features

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

Example:

```python
def test_detector_initialization():
    # Arrange
    model_name = 'yolov8n'
    
    # Act
    detector = HumanDetector(model=model_name)
    
    # Assert
    assert detector.model_name == model_name
    assert detector.model is not None
```

## Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
   
   Follow commit message conventions:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring
   - `perf:` for performance improvements

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe your changes in detail
   - Include screenshots for UI changes
   - Ensure CI passes

### PR Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged
- Your contribution will be credited

## Issue Guidelines

### Before Creating an Issue

- Search existing issues
- Check if it's already fixed in `main`
- Gather all relevant information

### Issue Types

#### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.10]
- CUDA: [e.g. 11.8]
- GPU: [e.g. RTX 3080]

**Additional Context**
Any other relevant information
```

#### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of what you want

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other context or screenshots
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions
- `refactor/` - Code refactoring

Example: `feature/add-bytetrack-algorithm`

### Commit Messages

Write clear, concise commit messages:

```
feat: add ByteTrack tracking algorithm

- Implement ByteTrack for crowded scenes
- Add configuration options
- Include unit tests
- Update documentation

Closes #123
```

## Community

- Join discussions in Issues and PRs
- Help answer questions
- Review pull requests
- Share your use cases

## Questions?

If you have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Acknowledgments

Thank you to all contributors who help make this project better!

---

**Happy Contributing! 🎉**
