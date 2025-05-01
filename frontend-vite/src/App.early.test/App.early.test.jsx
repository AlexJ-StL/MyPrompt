import React from 'react'
import App from '../App';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import "@testing-library/jest-dom";

// Mock the fetch API
global.fetch = jest.fn();

describe('App() App method', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  describe('Happy Paths', () => {
    test('renders the App component with initial state', () => {
      // Render the App component
      render(<App />);

      // Check if the initial elements are rendered correctly
      expect(screen.getByText('MyPrompt')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter your natural language request here...')).toBeInTheDocument();
      expect(screen.getByText('Optimize Prompt')).toBeInTheDocument();
    });

    test('optimizes prompt successfully', async () => {
      // Mock a successful fetch response
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ optimized_prompt: '<optimized>XML</optimized>' }),
      });

      // Render the App component
      render(<App />);

      // Simulate user input
      const textarea = screen.getByPlaceholderText('Enter your natural language request here...');
      fireEvent.change(textarea, { target: { value: 'Test request' } });

      // Simulate button click
      const button = screen.getByText('Optimize Prompt');
      fireEvent.click(button);

      // Wait for the optimized prompt to appear
      await waitFor(() => {
        expect(screen.getByText('Optimized XML Prompt:')).toBeInTheDocument();
        expect(screen.getByText('<optimized>XML</optimized>')).toBeInTheDocument();
      });
    });
  });

  describe('Edge Cases', () => {
    test('handles fetch error gracefully', async () => {
      // Mock a failed fetch response
      fetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ error: 'Fetch error' }),
      });

      // Render the App component
      render(<App />);

      // Simulate user input
      const textarea = screen.getByPlaceholderText('Enter your natural language request here...');
      fireEvent.change(textarea, { target: { value: 'Test request' } });

      // Simulate button click
      const button = screen.getByText('Optimize Prompt');
      fireEvent.click(button);

      // Wait for the error message to appear
      await waitFor(() => {
        expect(screen.getByText('Error: Fetch error')).toBeInTheDocument();
      });
    });

    test('disables input and button while loading', async () => {
      // Mock a delayed fetch response
      fetch.mockImplementationOnce(() =>
        new Promise((resolve) => setTimeout(() => resolve({ ok: true, json: async () => ({ optimized_prompt: '<optimized>XML</optimized>' }) }), 100))
      );

      // Render the App component
      render(<App />);

      // Simulate user input
      const textarea = screen.getByPlaceholderText('Enter your natural language request here...');
      fireEvent.change(textarea, { target: { value: 'Test request' } });

      // Simulate button click
      const button = screen.getByText('Optimize Prompt');
      fireEvent.click(button);

      // Check if the button and textarea are disabled
      expect(button).toBeDisabled();
      expect(textarea).toBeDisabled();

      // Wait for the loading to finish
      await waitFor(() => {
        expect(button).not.toBeDisabled();
        expect(textarea).not.toBeDisabled();
      });
    });
  });
});