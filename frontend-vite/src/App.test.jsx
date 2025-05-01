import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from './App';

describe('App', () => {
  it('renders the main heading', () => {
    render(<App />);
    expect(screen.getByText('MyPrompt')).toBeInTheDocument();
  });

  // TODO: Add more comprehensive tests for input, button click, and output display
});